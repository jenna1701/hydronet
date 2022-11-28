# Pytorch Geometric Implementation of SchNet

## Conda Environment

#### Using pip with conda
Not all packages are available with conda. To correctly direct a pip install in a conda environment, first `conda install pip`. Pip will install in your anaconda (or conda or miniconda) directory under the name of your environment (something like `/anaconda/envs/env_name/`). In all subsequent pip installs, replace `pip` with `/anaconda/envs/env_name/bin/pip`.

#### Pytorch 1.9.0 with cuda 11.1
This installation was used for training across NVIDIA P100s and RTX 2080 Ti GPUs.
```
conda install pytorch==1.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install pyg -c pyg
conda install -c conda-forge tensorboard ase fair-research-login h5py tqdm gdown
```

Note that it may be necessary to downgrade setuptools if tensorboard throws an error:
```
pip install setuptools==59.5.0
```

#### Pytorch 1.12.0 with cuda 11.3
This installation was used for training across NVIDIA A100s.
```
conda install pytorch==1.12.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install torch-scatter torch-sparse torch-cluster torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
conda install -c conda-forge tensorboard ase fair-research-login h5py tqdm gdown
```
Note that installing `torch-spine-conv` will likely produce a GLIBC error. It is safe to `pip uninstall torch-spine-conv` if the error occurs.

## Data Preprocessing
The HydroNet water cluster minima (`ID='min'`) and QM9 (`ID='qm9'`) datasets can be downloaded and preprocessed with the following call:

`python preprocess.py --sample $ID`

## Training 
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
    "load_model" (bool): flag to load architecture of pretrained model,
    "load_state" (bool): flag to load weights of pretrained model (if false, model is randomly initialized),
    "num_features" (int): layer feature size (only used if load_model: false),
    "num_interactions" (int): number of interaction layers (only used if load_model: false),
    "num_gaussians" (int): length of Gaussian basis (only used if load_model: false),
    "cutoff" (float): nearest neighbor cutoff distance (only used if load_model: false),
    "clip_value" (float): value for gradient clipping,
    "start_lr" (float): initial learning rate,
    "loss_fn" (str): flag for loss function (only "mse" currently implemented)
```


Training is run with the following call:

`python train_direct.py --savedir './test_train' --args 'train_args.json'`

## Inference
Inference on the test set is run with the following call:

`python test_set_errors.py --savedir './test_train'`

A csv will be saved as `test_set_inference.csv` with the actual and predicted properties for each sample.
