import os
import numpy as np

def create_init_split(args, suffix):
    cluster_list = np.load(args.cluster_path)
    np.random.shuffle(cluster_list)
    train_idx = cluster_list[0:args.n_train]
    val_idx = cluster_list[args.n_train:(args.n_train+args.n_val)]
    examine_idx = cluster_list[(args.n_train+args.n_val):(args.n_train+args.n_val+args.n_examine)]
    test_idx = cluster_list[(args.n_train+args.n_val+args.n_examine):]
    np.savez(os.path.join(args.savedir, f'split_00_{suffix}.npz'),
             train_idx=train_idx, 
             examine_idx=examine_idx,
             val_idx=val_idx,
             test_idx=test_idx)

