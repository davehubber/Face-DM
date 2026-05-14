#!/bin/bash
#
#SBATCH --partition=gpu_min11gb_ext                         # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=gpu_min11gb_ext                               # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=decode_celeba_test                          # Job name
#SBATCH --output=slurm_%x.%j.out                            # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err                             # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)

python decode_celeba_test.py \
  --diffae-root /nas-ctm01/homes/dacordeiro/diffae \
  --checkpoint /nas-ctm01/homes/dacordeiro/Face-DM/ffhq256_autoenc/last.ckpt \
  --dataset-dir /nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings_celeba \
  --embeddings-file celeba_diffae_zsem.npy \
  --metadata-file celeba_diffae_zsem_metadata.csv \
  --index 0 \
  --out-dir /nas-ctm01/homes/dacordeiro/Face-DM/debug_decode_celeba_zsem \
  --t-inv 200 \
  --t-step 200 \
  --also-recompute-semantic
