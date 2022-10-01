from torch_geometric.data import DataLoader
from utils.datasets import PrepackedDataset, QM9DataSet, WaterMinimaDataSet
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', required=True, type=str, help="Sample name ['qm9' or 'min']")
    parser.add_argument('--working_dir', default='./data', type=str, help='Directory in which to store raw data')
    parser.add_argument('--cached_dir', default='./data/cached_dataset', type=str, help='Directory in which to store processed data')
    args = parser.parse_args()

    raw_dir = os.path.join(args.working_dir, args.sample)

    if not os.path.isdir(raw_dir):
        os.mkdir(raw_dir)
    
    if args.sample=='qm9': 
        data = QM9DataSet(sample=args.sample, root=raw_dir, pre_transform=None)
    elif args.sample=='min':
        data = WaterMinimaDataSet(sample=args.sample, root=raw_dir, pre_transform=None)
    else:
        raise ValueError(f"{args.sample} is not an implemented dataset")

    loader = DataLoader(data, batch_size=1)
    dataset = PrepackedDataset([loader], None, args.sample, directory=args.cached_dir, num_elements=len(data.atom_types), max_num_atoms=data.max_num_atoms)



