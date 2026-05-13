#!/bin/bash
#
#SBATCH --partition=gpu_min8gb_ext                         # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=gpu_min8gb_ext                               # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=encode_diffae_celeba                           # Job name
#SBATCH --output=slurm_%x.%j.out                            # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err                             # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)

python encode_celeba_diffae_for_identity_mlp.py \
  --diffae-root /nas-ctm01/homes/dacordeiro/diffae \
  --checkpoint /nas-ctm01/homes/dacordeiro/Face-DM/ffhq256_autoenc/last.ckpt \
  --image-root /nas-ctm01/datasets/public/celeba \
  --identity-file /nas-ctm01/homes/dacordeiro/Face-DM/identity_CelebA_png.txt \
  --out-dir /nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings_celeba \
  --min-images-per-identity 30 \
  --batch-size 1
