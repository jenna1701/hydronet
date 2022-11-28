import os.path as op
import os
import torch 
import pandas as pd
import numpy as np
import json
import argparse

# custom functions
from utils import data, models, infer

parser = argparse.ArgumentParser()
parser.add_argument('--savedir', type=str, required=True, help='Directory where training results are saved')
args = parser.parse_args()

# read in args
savedir = args.savedir
with open(op.join(args.savedir, 'args.json')) as f:
    args_dict = json.load(f)
    args = argparse.Namespace(**args_dict)
args.savedir = savedir
args.load_state = True
args.load_model = True

# check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load trained model
net = models.load_model(args, mode='eval', device=device, frozen=True)

# get predictions on test set for each dataset
df = pd.DataFrame()
for dataset in args.datasets:
    loader = data.test_dataloader(args, dataset=dataset)
    tmp = infer.infer(loader, net, forces=args.train_forces, device=device)
    tmp['dataset']=dataset
    df = pd.concat([df, tmp], ignore_index=True, sort=False)

df.to_csv(op.join(args.savedir, 'test_set_inference.csv'), index=False)

print('Inference complete!')

