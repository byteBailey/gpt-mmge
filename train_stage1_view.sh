#!/bin/bash
# Stage 1 (view encoder): Modality-View + 4-Channel Routing 编码器跨数据集对比预训练
# 区别于 train_stage1.sh:
#   --encoder_type view    启用新双流 encoder
#   --K_text / --K_image   text/image view 的 K-query 数 (总 K = K_text + K_image)
#   --router_hidden        4 通道路由 MLP 隐藏维度

source activate gpt

export CUDA_VISIBLE_DEVICES=2

python pretrain_stage1.py \
  --datasets movies toys grocery arts cd \
  --output_dir output_stage1_view/exp2_no_align \
  --encoder_type view \
  --K_text 2 \
  --K_image 2 \
  --router_hidden 128 \
  --num_epochs 10 \
  --batch_size 64 \
  --lr 1e-4 \
  --wd 0.05 \
  --warmup_epochs 1 \
  --tau 0.1 \
  --div_lambda 0.01 \
  --lambda_t 0.5 \
  --lambda_i 0.5 \
  --lambda_align 0.0 \
  --num_hops 2 \
  --max_neighbors 20 \
  --mm_num_layers 4 \
  --mm_num_heads 8 \
  --mm_hidden_dim 1024 \
  --proj_dim 256 \
  --dropout 0.1 \
  --eval_every 1 \
  --eval_k 5 \
  --val_subsample 1000 \
  --num_workers 4 \
  --amp bf16
