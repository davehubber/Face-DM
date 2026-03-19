#!/bin/bash
#
#SBATCH --partition=gpu_min12gb_ext            # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=gpu_min12gb_ext                  # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=lt_avg_flower_animal        # Job name
#SBATCH --output=slurm_%x.%j.out               # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err                # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)

python train.py --data_dir "data_embeddings" --run_name "lt_avg_flower_animal" --batch_size 2048 --epochs 150