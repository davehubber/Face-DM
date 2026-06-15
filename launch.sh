#!/bin/bash
#
#SBATCH --partition=gpu_min24gb_ext         # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=gpu_min24gb_ext               # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=compute_smiling_scores        # Job name
#SBATCH --output=slurm_%x.%j.out           # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err            # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)

python compute_smiling_scores.py
