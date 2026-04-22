#!/bin/bash
#
#SBATCH --partition=debug_8gb            # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=debug_8gb                  # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=lt_avg_pc3                 # Job name
#SBATCH --output=slurm_%x.%j.out              # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err               # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)

python train_semantic.py --run_name "lt_avg_pc3" --epochs 10 --batch_size 8192 --mixed_precision "no" --eval_sampling_stride 150
