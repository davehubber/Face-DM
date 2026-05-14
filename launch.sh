#!/bin/bash
#
#SBATCH --partition=gpu_min11gb_ext                         # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=gpu_min11gb_ext                               # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=avg_diffae_id                          # Job name
#SBATCH --output=slurm_%x.%j.out                            # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err                             # File containing STDERR output. If ommited, use STDOUT.

# Commands / scripts to run (e.g., python3 train.py)

python train_latent_id.py \
  --dataset_root "/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings_celeba" \
  --run_name "avg_diffae_id" \
  --identity_classifier_path "/nas-ctm01/homes/dacordeiro/Face-DM/experiments/celeba_200id_diffae_identity_mlp/best_model.pt" \
  --identity_loss_weight 0.01 \
  --celeba_pair_probability 0.25 \
  --training_normalization combined \
  --identity_min_alpha 0.05 \
  --train_samples_per_epoch 1000000 \
  --val_samples 100000 \
  --batch_size 8192 \
  --epochs 50 \
  --lr 3e-4 \
  --weight_decay 1e-2 \
  --mixed_precision "fp16"
