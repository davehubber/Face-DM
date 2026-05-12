#!/bin/bash
#
#SBATCH --partition=debug_8gb                         # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=debug_8gb                               # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=test_diffae                           # Job name
#SBATCH --output=slurm_%x.%j.out                            # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err                             # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)

python test_diffae.py --diffae-root ../diffae --checkpoint ffhq256_autoenc/last.ckpt --embeddings diffae_embeddings/ffhq256_diffae_zsem.npy --metadata diffae_embeddings/ffhq256_diffae_zsem_metadata.csv --out-report diffae_average_face_test_report.txt --num-pairs 1000
