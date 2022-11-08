import os
import torch
import os.path as op
from torch_geometric.data import DataLoader
from torch.utils.data import ConcatDataset
from utils.datasets import PrepackedDataset
import torch.distributed as dist
import sys


def init_dataloader(args,
                    ngpus_per_node,
                    split = '00', 
                    shuffle=True):
    """
    Returns train, val, and list of examine loaders
    """
    pin_memory = False if args.train_forces else True

    if not isinstance(args.datasets, list):
        args.datasets = [args.datasets]
        
    train_data = []
    val_data = []
    examine_data = []
    for ds in args.datasets:
        dataset = PrepackedDataset(None, 
                                   op.join(args.savedir,f'split_{split}_{ds}.npz'), 
                                   ds, 
                                   directory=args.datadir)
        train_data = dataset.load_data('train')
        val_data   = dataset.load_data('val')
        # train_data.append(dataset.load_data('train'))
        # val_data.append(dataset.load_data('val'))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)

    print("Train data size {:5d}".format(len(train_data)), flush=True)
    train_loader = DataLoader(train_data, 
                              batch_size=int(args.batch_size), 
                              shuffle=(train_sampler is None),
                              num_workers=1 , 
                              pin_memory=pin_memory,
                              sampler=train_sampler, 
                              drop_last=True)
    print("train_loader size {:5d}".format(len(train_loader.dataset)), flush = True)
    val_loader = DataLoader(val_data, 
                            batch_size=int(args.batch_size), 
                            shuffle=False,
                            num_workers=1,
                            pin_memory=pin_memory,
                            drop_last=True)
    # ngpus_per_node
    return train_loader, val_loader, train_sampler


def test_dataloader(args, 
                    dataset,
                    split = '00'
                    ):

    dataset = PrepackedDataset(None, 
                               op.join(args.savedir,f'split_{split}_{dataset}.npz'), 
                               dataset, 
                               directory=args.datadir)
    data = dataset.load_data('test')
    
    batch_size = args.batch_size if len(data) > args.batch_size else len(data)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    return loader


