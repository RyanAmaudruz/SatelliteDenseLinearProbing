#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=60:00:00
#SBATCH --output=rsi_seg_train_multiple.out
#SBATCH --job-name=rsi_seg_train_multiple
#SBATCH --exclude=gcn45,gcn59

# Execute program located in $HOME

source activate rsi_seg

srun python tools/train.py --load-from /gpfs/work5/0/prjs0790/data/modified_checkpoints/odin_run_2024-03-22_12-19_ckp0_MODIFIED.pth & pid1=$!
wait $pid1
srun python tools/train.py --load-from /gpfs/work5/0/prjs0790/data/modified_checkpoints/odin_run_2024-03-22_12-19_ckp1_MODIFIED.pth & pid2=$!
wait $pid2
srun python tools/train.py --load-from /gpfs/work5/0/prjs0790/data/modified_checkpoints/odin_run_2024-03-22_12-19_ckp2_MODIFIED.pth & pid3=$!
wait $pid3
srun python tools/train.py --load-from /gpfs/work5/0/prjs0790/data/modified_checkpoints/odin_run_2024-03-22_12-19_ckp3_MODIFIED.pth & pid4=$!
wait $pid4
srun python tools/train.py --load-from /gpfs/work5/0/prjs0790/data/modified_checkpoints/odin_run_2024-03-22_12-19_ckp4_MODIFIED.pth & pid5=$!
wait $pid5
