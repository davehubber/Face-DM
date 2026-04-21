#!/bin/bash
#
#SBATCH --partition=gpu_min8gb_ext            # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=gpu_min8gb_ext                  # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=lt_avg_mag_10                 # Job name
#SBATCH --output=slurm_%x.%j.out              # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err               # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)

python train_semantic.py --run_name "lt_avg_mag_10" --epochs 20 --batch_size 8192 --injected_mse 0.5 --simulated_sampling_steps 10 --max_timesteps 10
