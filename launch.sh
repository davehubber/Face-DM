#!/bin/bash
#
#SBATCH --partition=gpu_min12gb_ext   # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=gpu_min12gb_ext         # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=train_projector    # Job name
#SBATCH --output=slurm_%x.%j.out      # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err       # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)

python train_identity_projector.py --dataset-root "/nas-ctm01/homes/dacordeiro/Face-DM/encoded_ffhq256_semantic_split" --batch-size 512 --epochs 100 --lr 1e-3 --hidden-dim 1024 --num-layers 3 --loss "cosine" --device "cuda"
