
export CUDA_VISIBLE_DEVICES=5

for num_epochs in 10 15; do
  for lr in 1e-5 2e-5; do
    for num_queries in 1 2 4; do
      for num_hops in 2; do
        for sample_neighbor_size in 5 10; do

        output_dir="output_toys/exp1/ep${num_epochs}_lr${lr}_nq${num_queries}_hop${num_hops}_seq${sample_neighbor_size}"

        echo "====== epochs=${num_epochs} lr=${lr} queries=${num_queries} hops=${num_hops} sample_size=${sample_neighbor_size} ======"
        echo "output_dir: ${output_dir}"

        python train.py \
          --dataset toys \
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
