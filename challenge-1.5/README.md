# Pytorch Geometric Implementation of SchNet

## Example usage
Set training arguments in `train_args.json`:
```
    "parallel" (bool): flag for training over multiple GPUs
    "create_splits" (bool): flag to create new split file
    "n_train" (int): number of samples in training set
    "n_val" (int): number of samples in validaiton set 
    "splitdir" (str): path to directory with split files (used when "create_splits": false)
    "datadir" (str): path to directory with databases
    "train_forces" (bool): flag to include force predictions in the loss function
    "energy_coeff" (float): weighting for energy and force components of the loss function (used when "train_forces":true; 0 = only train on forces, 1 = only train on energy)
    "n_epochs" (int): maximum number of epochs to train,
    "batch_size" (int):  batch size,
    "datasets" (list(str)): list of databases to train over,
    "start_model" (str): path to pretrained model state_dict,
    "load_state" (bool): flag to load weights of pretrained model (if false, model is randomly initialized),
    "clip_value" (float): value for gradient clipping,
    "start_lr" (float): initial learning rate,
    "loss_fn" (str): flag for loss function (only "mse" currently implemented)
```


Training is run with the following call:

`python train_direct.py --savedir './test_train' --args 'train_args.json'
