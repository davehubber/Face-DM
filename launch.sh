#!/bin/bash
#
#SBATCH --partition=gpu_min11gb_ext      # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=gpu_min11gb_ext            # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=test_diffae     # Job name
#SBATCH --output=slurm_%x.%j.out         # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err          # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)

python test_diffae.py --diffae-root "/nas-ctm01/homes/dacordeiro/diffae/" --checkpoint "/nas-ctm01/homes/dacordeiro/Face-DM/ffhq256_autoenc/last.ckpt" --embeddings "/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem.npy" --metadata "/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem_metadata.csv" --out-report "diffae_average_face_test_l1_normalized_report.txt" --zscore-mean "/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem_mean.npy" --zscore-std "/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem_std.npy"
