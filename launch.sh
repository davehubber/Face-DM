#!/bin/bash
#
#SBATCH --partition=gpu_min24gb_ext                         # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=gpu_min24gb_ext                               # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=arcface_to_diffae_mlp_v1                           # Job name
#SBATCH --output=slurm_%x.%j.out                            # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err                             # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)

python train_arcface_to_diffae.py --diffae-embeddings diffae_embeddings/ffhq256_diffae_zsem.npy --diffae-metadata diffae_embeddings/ffhq256_diffae_zsem_metadata.csv --arcface-embeddings arcface_embeddings/Face-DM/ffhq256_deepface_arcface_retinaface_l2norm.npy --arcface-metadata arcface_embeddings/Face-DM/ffhq256_deepface_arcface_metadata.csv --run-name arcface_to_diffae_mlp_v1 --experiments-root experiments --batch-size 512 --epochs 200 --lr 1e-3 --compute-retrieval
