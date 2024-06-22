#!/bin/bash
#SBATCH --time=80:00:00
#SBATCH -N 1
# #SBATCH --nodelist=node436
#SBATCH --exclude=node430
#SBATCH --gres=gpu:1
# #SBATCH --ntasks-per-node=8

source activate rsi_seg_new

srun python -m torch.distributed.launch tools/train.py
