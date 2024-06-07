#!/bin/bash
#SBATCH --job-name=gf3d_l12
#SBATCH -N1
#SBATCH -n1
#SBATCH --cpus-per-task=40
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --output=/l/users/aidana.nurakhmetova/thesis/3DETR_2/3detr/slurms2/slurm-%N-%j.out
#SBATCH --error=/l/users/aidana.nurakhmetova/thesis/3DETR_2/3detr/slurms2/slurm-%N-%j.err
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --qos=gpu-8

python main.py \
--dataset_name scannet \
--max_epoch 1080 \
--enc_type masked \
--enc_dropout 0.3 \
--nqueries 256 \
--dec_nlayers 10 \
--base_lr 5.0e-4 \
--batchsize_per_gpu 8 \
--matcher_giou_cost 2 \
--matcher_cls_cost 1 \
--matcher_center_cost 0 \
--matcher_objectness_cost 0 \
--loss_giou_weight 1 \
--loss_no_object_weight 0.25 \
--run 20 \
--checkpoint_dir /l/users/aidana.nurakhmetova/thesis/3DETR_2/3detr/checkpoints/3DETR-m-FusionIII/run20_100%_masked

# --use_color \