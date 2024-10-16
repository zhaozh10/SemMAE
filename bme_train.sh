#!/usr/bin/env bash
#SBATCH -p bme_gpu2
#SBATCH --job-name=HPM
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 5-00:00:00

set -x

source activate hpm
nvidia-smi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \


python -m torch.distributed.launch --nproc_per_node 8 \
        --nnodes 1 --node_rank 0 \
        --use_env main_pretrain_setting3.py \
        --output_dir ${OUTPUT_DIR} --log_dir=${OUTPUT_DIR} \
        --batch_size 128 \
        --model mae_vit_base_patch16 \
        --norm_pix_loss \
        --mask_ratio 0.75 \
        --epochs 800 \
        --warmup_epochs 40 \
        --blr 1.5e-4 --weight_decay 0.05 \
        --setting 3 \
        --data_path ${DATA_DIR}