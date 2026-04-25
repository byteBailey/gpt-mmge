# v6 版本：
# k跳子图，batch内动态采样，q个learnable query，多模态图编码器的输入改为 文本和图片的细粒度特征
# 图中所有实体序列 + 一个img token + 文本描述 + 文本问题 + 选项列表

# graph_seq 模式下，sample_neighbor_size 控制图的节点数，max_neighbors 决定节点间保留多少边

export CUDA_VISIBLE_DEVICES=7

for num_epochs in 10; do
  for lr in 2e-5; do
    for num_queries in 4; do
      for num_hops in 2; do
        for sample_neighbor_size in 5; do

        output_dir="output_grocery_84k/exp1/ep${num_epochs}_lr${lr}_nq${num_queries}_hop${num_hops}_seq${sample_neighbor_size}"

        echo "====== epochs=${num_epochs} lr=${lr} queries=${num_queries} hops=${num_hops} sample_size=${sample_neighbor_size} ======"
        echo "output_dir: ${output_dir}"

        python train.py \
          --dataset grocery \
          --model_name graph_llm \
          --llm_model_name 7b \
          --output_dir "${output_dir}" \
          --num_epochs ${num_epochs} \
          --lr ${lr} \
          --patience 1 \
          --max_neighbors 20 \
          --num_hops ${num_hops} \
          --mm_num_layers 4 \
          --mm_num_heads 8 \
          --mm_hidden_dim 1024 \
          --num_queries ${num_queries} \
          --graph_seq \
          --sample_neighbor_size ${sample_neighbor_size}

        done
      done
    done
  done
done
