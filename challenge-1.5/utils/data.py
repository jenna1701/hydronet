import os
import os.path as op
from torch_geometric.data import DataListLoader, DataLoader
from torch.utils.data import ConcatDataset
from utils.water_dataset import PrepackedDataset, gc_WaterDataSet
import sys


def init_dataloader(args, 
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
        
        train_data.append(dataset.load_data('train'))
        val_data.append(dataset.load_data('val'))

    if args.parallel:    
        train_loader = DataListLoader(ConcatDataset(train_data), batch_size=args.batch_size, shuffle=shuffle, pin_memory=pin_memory)
        val_loader = DataListLoader(ConcatDataset(val_data), batch_size=args.batch_size, shuffle=shuffle, pin_memory=pin_memory)
    else:
        train_loader = DataLoader(ConcatDataset(train_data), batch_size=args.batch_size, shuffle=shuffle, pin_memory=pin_memory)
        val_loader = DataLoader(ConcatDataset(val_data), batch_size=args.batch_size, shuffle=shuffle, pin_memory=pin_memory)
    
    return train_loader, val_loader


def test_dataloader(model_cat, 
                    dataset,
                    batch = 256,
                    splitdir = './data/splits/',
                    datadir = './data/cached_dataset/'
                    ):

    if model_cat == 'ipu':
        sys.path.insert(0, '/qfs/projects/ecp_exalearn/designs/NN_TBPMC/IPU_trained_models/pnnl_sandbox/schnet')
        import transforms
        
        # transforms for IPU model
        transform = transforms.create_transform(mode = "knn", k = 28,
                                                cutoff = 6.0, use_padding = False,
                                                max_num_atoms = 75, max_num_edges = None,
                                                use_qm9_energy = False, use_standardized_energy = False)
        data = gc_WaterDataSet(sample=f'split_00_test_set_{dataset}', 
                               root='./',
                               split_file = op.join(splitdir,f'split_00_{dataset}.npz'),
                               h5py_file = op.join(datadir,f'{dataset}_data.hdf5'),
                               idx_type = 'test',
                               pre_transform=transform)
    else:
        dataset = PrepackedDataset(None, op.join(splitdir,f'split_00_{dataset}.npz'), dataset, 
                                   directory=datadir)
        data = dataset.load_data('test')
    
    print('done preprocessing')

    batch_size = batch if len(data) > batch else len(data)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    return loader


