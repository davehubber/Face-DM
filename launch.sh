#!/bin/bash
#
#SBATCH --partition=gpu_min12gb_ext            # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=gpu_min12gb_ext                  # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=avg_default_branched        # Job name
#SBATCH --output=slurm_%x.%j.out               # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err                # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)

python train_coupled.py --dataset_path "/nas-ctm01/datasets/public/Oxford102Flowers/jpg" --run_name "avg_default_branched" --train_samples_per_epoch 36000 --val_samples 5000 --batch_size 256
