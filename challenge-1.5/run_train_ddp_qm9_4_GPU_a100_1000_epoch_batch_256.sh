#!/bin/bash

#SBATCH -A ecp_exalearn 
#SBATCH -p a100 # a100_80_shared # 
#SBATCH -o slurm-ddp_4_gpu_qm9_a100_1000_epoch_batch_256-%A.out
#SBATCH -e slurm-ddp_4-gpu_qm9_a100_1000_epoch_batch_256-%A.err
#SBATCH -J train_schenet 
#SBATCH --gres=gpu:4
#SBATCH -t 2-00:00:00 

module unload openmpi/2.1.1
module purge
module load cuda/11.3 gcc/9.1.0
export PATH="/people/firo017/.local/bin:$PATH"
source /people/firo017/torch-geometry/dependencies/poetry-cache/virtualenvs/torch-geometric-testing-f1daMZxP-py3.9/bin/activate

# export CUDA_VISIBLE_DEVICES=0,1
# python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=2 train_direct_ddp.py --savedir './test_train_ddp_qm9_A100_2_GPU_25_epoch_batch_256_timing' --args 'train_args_qm9_4_GPU_25_epoch_batch_256.json'

# --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29401

# export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

torchrun --standalone --nnodes=1  --nproc_per_node=4  train_direct_ddp.py --savedir './test_train_ddp_qm9_A100_4_GPU_1000_epoch_batch_256_timing_1' --args 'train_args_qm9_4_GPU_1000_epoch_batch_256.json'
