import os.path as op
import os
import torch 
import pandas as pd
import numpy as np

# custom functions
from utils.data import dataloader
from utils.models import load_model

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--savedir', type=str, default="/qfs/projects/ecp_exalearn/designs/finetune_comparison/data/test_set_results/", help='dir to save csv output')
parser.add_argument('--start-model', type=str, required=True, help='path to saved model state dict')
parser.add_argument('--split-choice', type=str, default='min_and_nonmin', help='data split to analyze')
args = parser.parse_args()
args.load_state=True

net = load_model(args, model_cat='finetune', mode='eval', device='cpu', frozen=True)

torch.pi = torch.acos(torch.zeros(1)).item() * 2 

def force_magnitude_error(actual, pred):
    # ||f_hat|| - ||f||
    return torch.sub(torch.norm(pred, dim=1), torch.norm(actual, dim=1))

def force_angular_error(actual, pred):
    # cos^-1( f_hat/||f_hat|| • f/||f|| ) / pi
    # batched dot product obtained with torch.bmm(A.view(-1, 1, 3), B.view(-1, 3, 1))
    a = torch.norm(actual, dim=1)
    p = torch.norm(pred, dim=1)

    return torch.div(torch.acos(torch.bmm(torch.div(actual.T, a).T.view(-1, 1, 3), 
                                          torch.div(pred.T, p).T.view(-1, 3, 1)).view(-1)), torch.pi)

def infer_with_error(loader, net):
    fme_mse, fme_mae = [],[]
    fae_mse, fae_mae = [],[]
    e_actual = []
    e_pred = []
    f_pred = []
    size = []
    for data in loader:
        # extract ground truth values
        e_actual += data.y.tolist()
        size += data.size.tolist()
        
        # get predicted values
        data.pos.requires_grad = True
        e = net(data)
        f = torch.autograd.grad(e, data.pos, grad_outputs=torch.ones_like(e), retain_graph=False)[0]
        
        # reshape f to ragged tensor
        # compute f errors for each sample
        start = 0
        for dsize in data.size.numpy()*3:
            f_ragged = f[start:start+dsize]
            f_act = data.f[start:start+dsize]
            
            fme = force_magnitude_error(f_act, f_ragged)
            fae = force_angular_error(f_act, f_ragged)
            
            fme_mae += [torch.mean(torch.abs(fme)).tolist()]
            fme_mse += [torch.mean(torch.square(fme)).tolist()]
            fae_mae += [torch.mean(torch.abs(fae)).tolist()]
            fae_mse += [torch.mean(torch.square(fae)).tolist()]
            
            start += dsize

        # get properties
        e_pred += e.tolist()
        

    return pd.DataFrame({'cluster_size': size, 
                         'e_actual': e_actual, 'e_pred': e_pred, 
                         'fme_mae': fme_mae, 'fae_mae': fae_mae,
                         'fme_mse': fme_mse, 'fae_mse': fae_mse})



df = pd.DataFrame()
for dataset in ['nonmin','min']:
    loader = dataloader(dataset, 'test', splitdir=f'./data/splits/{args.split_choice}/')
    tmp = infer_with_error(loader, net)
    tmp['dataset']=dataset

    df = pd.concat([df, tmp], ignore_index=True, sort=False)


saved_model = '-'.join(args.start_model.replace('.pt','').split('/')[-3:])

df['model']=args.start_model
df['model_cat']='ipu'

df.to_csv(op.join(args.savedir, f'finetune-{saved_model}-{args.split_choice}-split_00_test_set.csv'), index=False)

print(f'results saved to {args.savedir}/finetune-{saved_model}-{args.split_choice}-split_00_test_set.csv')

