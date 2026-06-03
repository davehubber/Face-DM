#!/bin/bash
#
#SBATCH --partition=gpu_min8gb_ext # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=gpu_min8gb_ext       # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=render_diffae_pca   # Job name
#SBATCH --output=slurm_%x.%j.out    # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err     # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)

CUDA_LAUNCH_BLOCKING=1 python render_diffae_pca.py
