#!/bin/bash
#
#SBATCH --partition=gpu_min24gb_ext            # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=gpu_min24gb_ext                  # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=avg_flowers_lpipsSorted     # Job name
#SBATCH --output=slurm_%x.%j.out               # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err                # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)

python train.py --dataset_path "/nas-ctm01/datasets/public/Oxford102Flowers/jpg" --run_name "avg_flowers_lpipsSorted" --partition_file "partition.csv" --batch_size 128 --epochs 150