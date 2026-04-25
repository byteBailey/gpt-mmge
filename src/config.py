import argparse


def parse_args_llama():
    parser = argparse.ArgumentParser(description="graph_llm")

    parser.add_argument("--model_name", type=str, default='graph_llm')
    parser.add_argument("--project", type=str, default="graph_prompt_tuning")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--dataset", type=str, default='grocery')
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--patience", type=float, default=30)
    parser.add_argument("--min_lr", type=float, default=5e-6)
    parser.add_argument("--resume", type=str, default='')

    # Model Training
    # parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--grad_steps", type=int, default=2)

    # Learning Rate Scheduler
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--warmup_epochs", type=float, default=1)

    # Inference
    parser.add_argument("--eval_batch_size", type=int, default=8)

    # LLM related
    parser.add_argument("--llm_model_name", type=str, default='7b')
    parser.add_argument("--llm_model_path", type=str, default='')
    parser.add_argument("--llm_frozen", type=str, default='True')
    parser.add_argument("--llm_num_virtual_tokens", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default='output')
    parser.add_argument("--max_txt_len", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=32)

    # llm adapter
    parser.add_argument("--adapter_len", type=int, default=10)
    parser.add_argument("--adapter_layer", type=int, default=30)

    # distributed training parameters
    # parser.add_argument("--log_dir", type=str, default='logs/')
    # parser.add_argument("--device", type=str, default='cuda')
    # parser.add_argument("--world_size", default=2, type=int, help="number of distributed processes")
    # parser.add_argument("--local_rank", default=-1, type=int)
    # parser.add_argument("--gpu", default='0,1', type=str)
    # parser.add_argument("--rank", default=-1, type=int)
    # parser.add_argument("--dist_on_itp", action="store_false")
    # parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    parser.add_argument("--log_dir", type=str, default='logs/')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_false")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    parser.add_argument("--num_workers", default=8, type=int)

    # GNN related
    parser.add_argument("--gnn_model_name", type=str, default='gat')
    parser.add_argument("--gnn_num_layers", type=int, default=4)
    parser.add_argument("--gnn_in_dim", type=int, default=1024)
    parser.add_argument("--gnn_hidden_dim", type=int, default=1024)
    parser.add_argument("--gnn_out_dim", type=int, default=1024)
    parser.add_argument("--gnn_num_heads", type=int, default=4)
    parser.add_argument("--gnn_dropout", type=float, default=0.0)

    # Multimodal Graph Encoder (used when graph has img_features/txt_features)
    parser.add_argument("--max_neighbors", type=int, default=20)
    parser.add_argument("--num_hops", type=int, default=2)
    parser.add_argument("--mm_num_layers", type=int, default=2)
    parser.add_argument("--mm_num_heads", type=int, default=8)
    parser.add_argument("--mm_hidden_dim", type=int, default=1024)
    parser.add_argument("--num_queries", type=int, default=1)

    # Graph sequence: LLaGA-style k-hop node expansion
    parser.add_argument("--graph_seq", action='store_true', default=False,
                        help='Expand k-hop neighbors as fixed-length token sequence for LLM')
    parser.add_argument("--sample_neighbor_size", type=int, default=10,
                        help='Number of neighbors sampled per node at each hop (graph_seq mode)')

    # Stage 2: 加载 Stage 1 预训练的多模态图编码器
    parser.add_argument("--stage1_ckpt", type=str, default='',
                        help='Stage 1 ckpt 路径; 空字符串表示走端到端 (随机初始化 MMGE)')
    parser.add_argument("--freeze_mm_encoder", action='store_true', default=False,
                        help='冻结多模态图编码器 (Stage 2 默认开启, 只训 projector)')

    # Encoder type: 选择多模态图编码器架构
    parser.add_argument("--encoder_type", type=str, default='mmge',
                        choices=['mmge', 'view'],
                        help="'mmge': 老 K-query 单流 encoder; "
                             "'view': Modality-view + 4-channel routing 双流 encoder (新方法)")
    parser.add_argument("--K_text", type=int, default=2,
                        help='view encoder 的 text-view query 数')
    parser.add_argument("--K_image", type=int, default=2,
                        help='view encoder 的 image-view query 数 (总 K = K_text + K_image)')
    parser.add_argument("--router_hidden", type=int, default=128,
                        help='4-channel routing MLP 隐藏维度')

    args = parser.parse_args()
    return args

