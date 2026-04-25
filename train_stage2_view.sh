#!/bin/bash
# Stage 2 (view encoder): 加载 Stage 1 view encoder ckpt + LLaMA (冻结), 仅训练投影层
# 区别于 train_stage2.sh:
#   --encoder_type view    必须与 Stage 1 训练时一致
#   --K_text / --K_image   架构超参, 必须与 Stage 1 ckpt 完全一致
#   --router_hidden        同上
#
# 注意: --mm_num_layers / --mm_num_heads / --mm_hidden_dim / --K_text / --K_image
#       必须与 Stage 1 训练时完全一致, 否则 load_state_dict 会报 shape mismatch.

source activate gpt

export CUDA_VISIBLE_DEVICES=1

# ── Stage 1 view ckpt 路径 (跑完 train_stage1_view.sh 后填入) ──
STAGE1_CKPT="/home/liyijun/gpt-mmge/gpt-mmge-v8/output_stage1_view/exp1/stage1_best.pt"

# ── Encoder type & architecture: 必须与 Stage 1 ckpt 完全一致 ──
ENCODER_TYPE="view"
K_TEXT=2
K_IMAGE=2
ROUTER_HIDDEN=128
MM_NUM_LAYERS=4
MM_NUM_HEADS=8
MM_HIDDEN_DIM=1024

# ── 数据采样 ──
NUM_HOPS=2
MAX_NEIGHBORS=20

# ── 训练超参 ──
NUM_EPOCHS=10

for dataset in movies toys arts cd grocery; do
  for lr in 5e-5; do
    for sample_neighbor_size in 5; do

      output_dir="output_stage2_view/exp1/${dataset}/ep${NUM_EPOCHS}_lr${lr}_seq${sample_neighbor_size}"

      echo "====== dataset=${dataset} lr=${lr} sample_size=${sample_neighbor_size} ======"
      echo "stage1_ckpt: ${STAGE1_CKPT}"
      echo "output_dir: ${output_dir}"

      python train.py \
        --dataset ${dataset} \
        --model_name graph_llm \
        --llm_model_name 7b \
        --output_dir "${output_dir}" \
        --stage1_ckpt "${STAGE1_CKPT}" \
        --freeze_mm_encoder \
        --encoder_type ${ENCODER_TYPE} \
        --K_text ${K_TEXT} \
        --K_image ${K_IMAGE} \
        --router_hidden ${ROUTER_HIDDEN} \
        --num_epochs ${NUM_EPOCHS} \
        --lr ${lr} \
        --patience 2 \
        --max_neighbors ${MAX_NEIGHBORS} \
        --num_hops ${NUM_HOPS} \
        --mm_num_layers ${MM_NUM_LAYERS} \
        --mm_num_heads ${MM_NUM_HEADS} \
        --mm_hidden_dim ${MM_HIDDEN_DIM} \
        --graph_seq \
        --sample_neighbor_size ${sample_neighbor_size}

    done
  done
done
