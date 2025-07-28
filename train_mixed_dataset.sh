#!/bin/bash
# 使用混合数据集训练VQ-CycleGAN的示例脚本

# 设置基本参数
DATAROOT="./datasets/xijing_split"  # 你的数据集路径
NAME="vq_cyclegan_mixed_training"   # 实验名称
MODEL="vq_cycle_gan"                # 模型类型
DATASET_MODE="mixed"                # 使用混合数据集

# 训练参数
BATCH_SIZE=4
N_EPOCHS=100
N_EPOCHS_DECAY=100
LR=0.0002

# 损失权重
LAMBDA_A=10.0
LAMBDA_B=10.0
LAMBDA_REC_A=10.0
LAMBDA_REC_B=10.0
LAMBDA_VQ=1.0
LAMBDA_PAIRED=20.0  # 配对损失权重

# VQ参数
N_EMBED=512
EMBED_DIM=256
BETA=0.25
DECAY=0.99

# 执行训练命令
python train.py \
    --dataroot $DATAROOT \
    --name $NAME \
    --model $MODEL \
    --dataset_mode $DATASET_MODE \
    --batch_size $BATCH_SIZE \
    --n_epochs $N_EPOCHS \
    --n_epochs_decay $N_EPOCHS_DECAY \
    --lr $LR \
    --lambda_A $LAMBDA_A \
    --lambda_B $LAMBDA_B \
    --lambda_rec_A $LAMBDA_REC_A \
    --lambda_rec_B $LAMBDA_REC_B \
    --lambda_vq $LAMBDA_VQ \
    --lambda_paired $LAMBDA_PAIRED \
    --n_embed $N_EMBED \
    --embed_dim $EMBED_DIM \
    --beta $BETA \
    --decay $DECAY \
    --display_freq 100 \
    --print_freq 50 \
    --save_latest_freq 1000 \
    --save_epoch_freq 10 \
    --gpu_ids 0

echo "Training completed!"
