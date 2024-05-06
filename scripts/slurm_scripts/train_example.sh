#!/bin/bash
# Pytorch Lightning training example script on SLURM

# PyTorch Lightning 提供了在 SLURM 集群上自动处理wall-time重新提交的功能。以后有兴趣可以看看
# https://lightning.ai/docs/pytorch/1.8.6/clouds/cluster_advanced.html#enable-auto-wall-time-resubmitions

#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4   # This needs to match Trainer(devices=...)
#SBATCH --cpus-per-task=8     # PL没有说明是否要设置cpus-per-task，但是尝试过设置此项有助于训练，后期可以再尝试验证
#SBATCH --mem=0
#SBATCH --time=0-02:00:00

# activate conda env
conda activate pldev

# debugging flags (optional) # debug用的
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# run script from above
srun python train.py