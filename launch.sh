#!/bin/bash
#
#SBATCH --partition=gpu_min8gb_ext                         # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=gpu_min8gb_ext                               # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=train_celeba_mlp                           # Job name
#SBATCH --output=slurm_%x.%j.out                            # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err                             # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)

python train_celeba_mlp.py \
  --dataset-dir /nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings_celeba \
  --out-dir /nas-ctm01/homes/dacordeiro/Face-DM/experiments/celeba_200id_diffae_identity_mlp \
  --epochs 100 \
  --batch-size 256 \
  --hidden-dim 512 \
  --dropout 0.20 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --patience 20
