#!/bin/bash
#
#SBATCH --partition=gpu_min24gb_ext                     # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=gpu_min24gb_ext                           # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=avg_diffae_pit_l1cos_10ts               # Job name
#SBATCH --output=slurm_%x.%j.out                        # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err                         # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)

python train_latent.py --run_name "avg_diffae_pit_l1cos_10ts" --max_timesteps 10 --batch_size 24576 --epochs 15 --cosine_loss_weight 0.1
