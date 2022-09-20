from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
from torch_geometric.nn.models.schnet import GaussianSmearing, \
    InteractionBlock, ShiftedSoftplus
from torch_scatter.scatter import scatter_add
from torch_geometric.nn import knn_graph
import sys
import logging
import argparse

def load_model(args, model_cat, mode='eval', device='cpu', frozen=False):
    """
    Load trained model for eval
    model_cat = ['ipu', 'finetune', 'multifi']
    """
    
    if model_cat in ['ipu','finetune']:
        net = load_pretrained_model(args, model_cat, device=device, frozen=frozen)
    else:
        net = MultiFiSchNet(args, device = device)
    
    if mode=='eval':
        # set to eval mode
        net.eval()
        print('model set to eval')

    return net

def load_pretrained_model(args, model_cat='', device='cpu', frozen=False):
    """
    Load single SchNet model
    """
    device = torch.device(device)
    
    # load state dict of trained model
    state=torch.load(args.start_model)
    
    # extract model params from model state dict
    num_gaussians = state['basis_expansion.offset'].shape[0]
    num_filters = state['interactions.0.mlp.0.weight'].shape[0]
    num_interactions = len([key for key in state.keys() if '.lin.bias' in key])
    
    # load model architecture
    net = SchNet(num_features = num_filters,
                 num_interactions = num_interactions,
                 num_gaussians = num_gaussians,
                 cutoff = 6.0)
    
    logging.info(f'model loaded from {args.start_model}')
    
    if args.load_state:
        # load trained weights into model
        net.load_state_dict(state)
        logging.info('model weights loaded')
    
    net.to(device)
    
    #register backward hook --> gradient clipping
    if not frozen:
        for p in net.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -args.clip_value, args.clip_value))

    return net


class SchNet(nn.Module):
    def __init__(self,
                 num_features: int = 100,
                 num_interactions: int = 4,
                 num_gaussians: int = 25,
                 cutoff: float = 6.0,
                 batch_size: Optional[int] = None):
        """
        :param num_features (int): The number of hidden features used by both
            the atomic embedding and the convolutional filters (default: 128).
        :param num_interactions (int): The number of interaction blocks
            (default: 2).
        :param num_gaussians (int): The number of gaussians used in the radial
            basis expansion (default: 50).
        :param cutoff (float): Cutoff distance for interatomic interactions
            which must match the one used to build the radius graphs
            (default: 6.0).
        :param batch_size (int, optional): The number of molecules in the batch.
            This can be inferred from the batch input when not supplied.
        """
        super().__init__()
        self.num_features = num_features
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.batch_size = batch_size

        self.atom_embedding = nn.Embedding(100,
                                           self.num_features,
                                           padding_idx=0)
        self.basis_expansion = GaussianSmearing(0.0, self.cutoff,
                                                self.num_gaussians)

        self.interactions = nn.ModuleList()

        for _ in range(self.num_interactions):
            block = InteractionBlock(self.num_features, self.num_gaussians,
                                     self.num_features, self.cutoff)
            self.interactions.append(block)

        self.lin1 = nn.Linear(self.num_features, self.num_features // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = nn.Linear(self.num_features // 2, 1)

        self.reset_parameters()

    def hyperparameters(self):
        """
        hyperparameters for the SchNet model.

        :returns: Dictionary of hyperparamters.
        """
        return {
            "num_features": self.num_features,
            "num_interactions": self.num_interactions,
            "num_gaussians": self.num_gaussians,
            "cutoff": self.cutoff,
            "batch_size": self.batch_size
        }

    def extra_repr(self) -> str:
        """
        extra representation for the SchNet model.

        :returns: comma-separated string of the model hyperparameters.
        """
        s = []
        for key, value in self.hyperparameters().items():
            s.append(f"{key}={value}")

        return ", ".join(s)

    def reset_parameters(self):
        """
        Initialize learnable parameters used in training the SchNet model.
        """
        self.atom_embedding.reset_parameters()

        for interaction in self.interactions:
            interaction.reset_parameters()

        xavier_uniform_(self.lin1.weight)
        zeros_(self.lin1.bias)
        xavier_uniform_(self.lin2.weight)
        zeros_(self.lin2.bias)

    def forward(self, data):
        """
        Forward pass of the SchNet model

        :param z: Tensor containing the atomic numbers for each atom in the
            batch. Vector with size [num_atoms].
        :param edge_weight: Tensor containing the interatomic distances for each
            interacting pair of atoms in the batch. Vector with size [num_edges]
        :param edge_index: Tensor containing the indices defining the
            interacting pairs of atoms in the batch. Matrix with size
            [2, num_edges]
        :param batch: Tensor assigning each atom within a batch to a molecule.
            This is used to perform per-molecule aggregation to calculate the
            predicted energy. Vector with size [num_atoms]
        :param energy_target (optional): Tensor containing the energy target to
            use for evaluating the mean-squared-error loss when training.
        """
        # Collapse any leading batching dimensions
        pos = data.pos
        edge_index = knn_graph(
            data.pos,
            28,
            data.batch,
            loop=False,
        )

        row, col = edge_index

        edge_weight = torch.norm(pos[col] - pos[row], p=2, dim=-1).view(-1, 1)
        edge_index = edge_index.view(2, -1).long()
        batch = data.batch.long()

        h = self.atom_embedding(data.z.long())
        edge_attr = self.basis_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        mask = (data.z == 0).view(-1, 1)
        h = h.masked_fill(mask.expand_as(h), 0.)

        batch = batch.view(-1)
        out = scatter_add(h, batch, dim=0, dim_size=self.batch_size).view(-1)

        return out

    @staticmethod
    def loss(input, target):
        """
        Calculates the mean squared error

        This loss assumes that zeros are used as padding on the target so that
        the count can be derived from the number of non-zero elements.
        """
        loss = F.mse_loss(input, target, reduction="sum")
        N = (target != 0.0).to(loss.dtype).sum()
        loss = loss / N
        return identity_loss(loss, reduction="none")



class MultiFiSchNet(nn.Module):
    def __init__(self,
                 args,
                 device = 'cpu'):
        """
        :param model_path (str): Path to trained model
        :param device (str): Device to run model on ['cpu', 'cuda']
        """
        super().__init__()
        
        self.args = args
        
        # load pretrained model
        self.lowfi_model = load_pretrained_model(args, model_cat='finetune', frozen=True, device=device)

        # freeze lowfi model layers
        for param in self.lowfi_model.parameters():
            param.requires_grad = False
            
        # load empty model with smaller architecture
        state=torch.load(args.start_model, map_location=torch.device(device))
        num_gaussians = state['basis_expansion.offset'].shape[0]
        num_filters = state['interactions.0.mlp.0.weight'].shape[0]
        num_interactions = len([key for key in state.keys() if '.lin.bias' in key])
        
        self.dif_model = SchNet(num_features = int(num_filters/2),
                     num_interactions = int((num_interactions+1)/2),
                     num_gaussians = num_gaussians,
                     cutoff = 6.0)
        
        self.dif_model.to(device)
        
        for p in self.dif_model.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -args.clip_value, args.clip_value))

        # correlation/sum layer 
        self.correlation = nn.Linear(1, 1, bias=False, device=device)
        
        s = self.correlation.state_dict()
        for k in s.keys():
            s[k] = torch.ones_like(s[k])
        self.correlation.load_state_dict(s)
        
        for p in self.correlation.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -args.clip_value, args.clip_value))


    def forward(self, data):
        """
        Forward pass of the SchNet model
        :param data: data from data loader
        """     
        
        y_low = self.correlation(self.lowfi_model(data).view(-1,1)).T[0]
        y_dif = self.dif_model(data)
        
        # interleave y_low and y_dif
        y = torch.stack((y_low,y_dif), dim=1).view(-1,2)
        y = torch.sum(y, dim=1)
        
        return y
