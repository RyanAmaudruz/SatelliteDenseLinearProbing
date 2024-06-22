#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=20:00:00
#SBATCH --output=rsi_seg_train.out
#SBATCH --job-name=rsi_seg_train
#SBATCH --exclude=gcn45,gcn59

# Execute program located in $HOME

source activate rsi_seg

srun python -m torch.distributed.launch tools/train.py
