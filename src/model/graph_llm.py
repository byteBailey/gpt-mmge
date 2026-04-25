import contextlib
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    # prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)


from src.model.gnn import load_gnn_model
from src.model.multimodal_graph_encoder import MultimodalGraphEncoder, Projector
from src.model.modality_view_encoder import ModalityViewGraphEncoder
from src.dataset.subgraph_builder import SubgraphBuilder


ignore_index = -100


class GraphLLM(torch.nn.Module):

    def __init__(
        self,
        graph,
        graph_type,
        prompt,
        args,
    ):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens

        print('Loading LLAMA')
        kwargs = {
            "max_memory": {0: '80GiB'},
            "device_map": "auto",
            "revision": "main",
        }

        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False, revision=kwargs["revision"])
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'

        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            **kwargs
        )

        if args.llm_frozen == 'True':
            print("Freezing LLAMA!")
            for name, param in model.named_parameters():
                param.requires_grad = False
        else:
            print("Training LLAMA with LORA!")
            # model = prepare_model_for_int8_training(model)
            model = prepare_model_for_kbit_training(model)
            lora_r: int = 8
            lora_alpha: int = 16
            lora_dropout: float = 0.05
            lora_target_modules = [
                "q_proj",
                "v_proj",
            ]
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

        self.model = model
        print('Finish loading LLAMA!')

        llm_hidden_size = getattr(self.model.config, 'hidden_size', None)
        if llm_hidden_size is None:
            raise ValueError('The loaded LLM config must expose hidden_size.')

        # ── 图编码器: 根据图数据自动选择 ──
        self.use_mm_encoder = hasattr(graph, 'img_features') and graph.img_features is not None

        if self.use_mm_encoder:
            print('Using Multimodal Graph Encoder')
            mm_hidden_dim = getattr(args, 'mm_hidden_dim', args.gnn_hidden_dim)
            mm_num_layers = getattr(args, 'mm_num_layers', 2)
            mm_num_heads = getattr(args, 'mm_num_heads', 8)
            max_neighbors = getattr(args, 'max_neighbors', 20)
            num_queries = getattr(args, 'num_queries', 1)

            # 从图数据自动检测特征维度
            img_dim = graph.img_features.shape[-1]
            txt_dim = graph.txt_features.shape[-1]
            print(f'  img_dim={img_dim}, txt_dim={txt_dim}, num_queries={num_queries}')

            encoder_type = getattr(args, 'encoder_type', 'mmge')
            if encoder_type == 'view':
                K_text = getattr(args, 'K_text', 2)
                K_image = getattr(args, 'K_image', 2)
                router_hidden = getattr(args, 'router_hidden', 128)
                print(f'  encoder_type=view (Modality-view + 4-Channel Routing), '
                      f'K_text={K_text}, K_image={K_image}, router_hidden={router_hidden}')
                self.mm_encoder = ModalityViewGraphEncoder(
                    img_dim=img_dim,
                    txt_dim=txt_dim,
                    hidden_dim=mm_hidden_dim,
                    num_layers=mm_num_layers,
                    num_heads=mm_num_heads,
                    K_text=K_text,
                    K_image=K_image,
                    router_hidden=router_hidden,
                    dropout=args.gnn_dropout,
                ).to(self.model.device)
            else:
                print(f'  encoder_type=mmge (legacy K-query single-stream)')
                self.mm_encoder = MultimodalGraphEncoder(
                    img_dim=img_dim,
                    txt_dim=txt_dim,
                    hidden_dim=mm_hidden_dim,
                    num_layers=mm_num_layers,
                    num_heads=mm_num_heads,
                    num_relations=1,
                    num_queries=num_queries,
                    dropout=args.gnn_dropout,
                ).to(self.model.device)

            # ── Stage 2: 加载 Stage 1 预训练权重 (可选) ──
            stage1_ckpt = getattr(args, 'stage1_ckpt', '')
            if stage1_ckpt:
                print(f'Loading Stage 1 ckpt: {stage1_ckpt}')
                ckpt = torch.load(stage1_ckpt, map_location='cpu', weights_only=False)
                # ckpt['model'] 形如 {'encoder.xxx': ..., 'proj_head.xxx': ...}
                # 只取 encoder.* (proj_head 是 Stage 1 对比头, Stage 2 不用)
                enc_state = {k[len('encoder.'):]: v
                             for k, v in ckpt['model'].items()
                             if k.startswith('encoder.')}
                missing, unexpected = self.mm_encoder.load_state_dict(enc_state, strict=True)
                print(f'  loaded encoder.* params={len(enc_state)} '
                      f'(missing={len(missing)}, unexpected={len(unexpected)})')

            # ── 冻结 MMGE (Stage 2 默认开启) ──
            self.freeze_mm_encoder = getattr(args, 'freeze_mm_encoder', False)
            if self.freeze_mm_encoder:
                for p in self.mm_encoder.parameters():
                    p.requires_grad = False
                self.mm_encoder.eval()
                print('MMGE frozen (requires_grad=False, eval mode)')

            self.mm_projector = Projector(
                encoder_dim=mm_hidden_dim,
                llm_dim=llm_hidden_size,
            ).to(self.model.device)

            # 图片特征直接投影到 LLM 空间 (使用 MAGB 单向量 CLS 特征, 不使用细粒度 patch)
            self.img_projector = Projector(
                encoder_dim=graph.img_features.shape[-1],
                llm_dim=llm_hidden_size,
            ).to(self.model.device)

            num_hops = getattr(args, 'num_hops', 2)
            self.subgraph_builder = SubgraphBuilder(graph, max_neighbors=max_neighbors, num_hops=num_hops)

            # Graph sequence mode (LLaGA-style node expansion)
            self.graph_seq = getattr(args, 'graph_seq', False)
            self.sample_neighbor_size = getattr(args, 'sample_neighbor_size', 10)
            if self.graph_seq:
                seq_len = sum(self.sample_neighbor_size ** i for i in range(num_hops + 1))
                print(f'  graph_seq=True, sample_neighbor_size={self.sample_neighbor_size}, '
                      f'num_hops={num_hops}, seq_len={seq_len} (x num_queries={num_queries})')

            # 保留原始图片特征引用, 用于直接索引中心节点的单向量 CLS 特征
            self.graph_img_features = graph.img_features  # [N, P, img_dim]
        else:
            print('Using legacy GAT encoder')
            self.graph_encoder = load_gnn_model[args.gnn_model_name](
                in_channels=graph.x.shape[-1],
                hidden_channels=args.gnn_hidden_dim,
                out_channels=args.gnn_out_dim,
                num_layers=args.gnn_num_layers,
                dropout=args.gnn_dropout,
                num_heads=args.gnn_num_heads,
            ).to(self.model.device)

            self.projector = nn.Sequential(
                nn.Linear(args.gnn_out_dim, 2048),
                nn.Sigmoid(),
                nn.Linear(2048, llm_hidden_size),
            ).to(self.model.device)

        self.word_embedding = self.model.model.get_input_embeddings()

    def train(self, mode: bool = True):
        # 保持冻结的 MMGE 在 eval 模式 (关掉 dropout); LLM 已被 hf 内部固定
        super().train(mode)
        if getattr(self, 'use_mm_encoder', False) and getattr(self, 'freeze_mm_encoder', False):
            self.mm_encoder.eval()
        return self

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def encode_graphs(self, samples):
        """
        Returns:
            graph_embeds: [B, T, llm_hidden_size]  T = K (default) or seq_len*K (graph_seq)
            img_embeds:   [B, llm_hidden_size] (MM path) or None (legacy)
        """
        if self.use_mm_encoder:
            center_ids = samples['id']
            device = self.model.device

            if self.graph_seq:
                # ── LLaGA-style: 固定长度节点序列 ──
                batched, batch_vec, center_indices, hop_seqs = \
                    self.subgraph_builder.build_batch_seq(center_ids, self.sample_neighbor_size)
                batched = {k: v.to(device) for k, v in batched.items()}
                hop_seqs = hop_seqs.to(device)              # [B, seq_len]

                entity_repr = self.mm_encoder(**batched)     # [M_total, K, hidden_dim]
                B, seq_len = hop_seqs.shape
                K = entity_repr.size(1)

                # 用 hop_seq 从 entity_repr 中取序列节点表示, PAD 位置置零
                pad_mask = (hop_seqs == -1)                  # [B, seq_len]
                safe_idx = hop_seqs.clamp(min=0)             # PAD→0 用于安全索引
                seq_repr = entity_repr[safe_idx.view(-1)]    # [B*seq_len, K, D]
                seq_repr = seq_repr.view(B, seq_len, K, -1)  # [B, seq_len, K, D]
                # PAD 位置清零 (用 masked_fill 避免 in-place 修改破坏 autograd)
                seq_repr = seq_repr.masked_fill(
                    pad_mask.unsqueeze(-1).unsqueeze(-1), 0.0)  # [B,seq_len,1,1] broadcast

                # 合并 K 维: K=1 时 squeeze, K>1 时 mean pool 为 [B, seq_len, D]
                if K == 1:
                    seq_repr = seq_repr.squeeze(2)           # [B, seq_len, D]
                else:
                    seq_repr = seq_repr.mean(dim=2)          # [B, seq_len, D]

                graph_tokens = self.mm_projector(seq_repr)   # [B, seq_len, llm_hidden_size]
            else:
                # ── 原始方式: 仅中心节点的 K 个 query ──
                batched, batch_vec, center_indices = self.subgraph_builder.build_batch(center_ids)
                batched = {k: v.to(device) for k, v in batched.items()}
                center_indices = center_indices.to(device)

                entity_repr = self.mm_encoder(**batched)     # [M_total, K, hidden_dim]
                center_repr = entity_repr[center_indices]    # [B, K, hidden_dim]
                graph_tokens = self.mm_projector(center_repr)  # [B, K, llm_hidden_size]

            # 中心节点的原始图片 CLS 特征 → img_projector → img_tokens
            center_ids_t = torch.tensor(center_ids, dtype=torch.long)
            img_raw = self.graph_img_features[center_ids_t]    # [B, P, img_dim]
            img_raw = img_raw.mean(dim=1).to(device)           # [B, img_dim]
            img_tokens = self.img_projector(img_raw)           # [B, llm_hidden_size]

            return graph_tokens, img_tokens
        else:
            x = samples['x'].to(self.model.device)
            edge_index = samples['edge_index'].to(self.model.device)
            mapping = samples['mapping'].to(self.model.device)

            n_embeds, _ = self.graph_encoder(x, edge_index)
            inputs_embeds = self.projector(n_embeds[mapping])  # [B, llm_hidden_size]
            return inputs_embeds.unsqueeze(1), None            # [B, 1, llm_hidden_size], None

    def _get_prefix_embeds(self, device):
        """返回结构化 prefix 的 embedding，用于包裹 graph/img 虚拟 token。"""
        p1_ids = self.tokenizer("The graph features of the product ", add_special_tokens=False).input_ids
        p2_ids = self.tokenizer(", the image feature of the product ", add_special_tokens=False).input_ids
        p3_ids = self.tokenizer(", ", add_special_tokens=False).input_ids
        p1 = self.word_embedding(torch.tensor(p1_ids, device=device))  # [L1, D]
        p2 = self.word_embedding(torch.tensor(p2_ids, device=device))  # [L2, D]
        p3 = self.word_embedding(torch.tensor(p3_ids, device=device))  # [L3, D]
        return p1, p2, p3

    def forward(self, samples):

        # encode description, questions and labels
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        desriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
        labels = self.tokenizer(samples["label"], add_special_tokens=False)
        device = self.model.device
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(device)).unsqueeze(0)
        bos_embeds = self.word_embedding(torch.tensor(self.tokenizer.bos_token_id).to(device)).unsqueeze(0)

        # encode graphs → [B, K, D], [B, D] or None
        graph_embeds, img_embeds = self.encode_graphs(samples)

        # prefix embeddings (仅 MM 路径)
        if img_embeds is not None:
            p1, p2, p3 = self._get_prefix_embeds(device)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        for i in range(batch_size):
            # Add bos & eos token
            label_input_ids = labels.input_ids[i] + [self.tokenizer.eos_token_id]
            input_ids = desriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + label_input_ids
            text_embeds = self.word_embedding(torch.tensor(input_ids).to(device))

            if img_embeds is not None:
                # [BOS] "The graph features of the product " [G1]...[GK] ", the image feature of the product " [IMG] ", " [desc+question+label+EOS]
                inputs_embeds = torch.cat([
                    bos_embeds,                             # [1, D]
                    p1,                                     # "The graph features of the product "
                    graph_embeds[i],                        # [K, D]  K graph tokens
                    p2,                                     # ", the image feature of the product "
                    img_embeds[i].unsqueeze(0),             # [1, D]  image token
                    p3,                                     # ", "
                    text_embeds,                            # desc + question + label + EOS
                ], dim=0)
            else:
                # legacy: [BOS] [GRAPH] [desc+question+label+EOS]
                inputs_embeds = torch.cat([bos_embeds, graph_embeds[i], text_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [ignore_index] * (inputs_embeds.shape[0]-len(label_input_ids))+label_input_ids
            batch_label_input_ids.append(label_input_ids)

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]
            batch_label_input_ids[i] = [ignore_index] * pad_length+batch_label_input_ids[i]
        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(device)
        attention_mask = torch.tensor(batch_attention_mask).to(device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(device)

        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )

        return outputs.loss

    def inference(self, samples):
        device = self.model.device
        # encode description and questions
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        desriptions = self.tokenizer(samples["desc"], add_special_tokens=False)

        # encode special tokens
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(device)).unsqueeze(0)
        bos_embeds = self.word_embedding(torch.tensor(self.tokenizer.bos_token_id).to(device)).unsqueeze(0)

        # encode graphs → [B, K, D], [B, D] or None
        graph_embeds, img_embeds = self.encode_graphs(samples)

        # prefix embeddings (仅 MM 路径)
        if img_embeds is not None:
            p1, p2, p3 = self._get_prefix_embeds(device)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            input_ids = desriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i]
            text_embeds = self.word_embedding(torch.tensor(input_ids).to(device))

            if img_embeds is not None:
                # [BOS] "The graph features of the product " [G1]...[GK] ", the image feature of the product " [IMG] ", " [desc+question]
                inputs_embeds = torch.cat([
                    bos_embeds,
                    p1,
                    graph_embeds[i],                        # [K, D]
                    p2,
                    img_embeds[i].unsqueeze(0),             # [1, D]
                    p3,
                    text_embeds,
                ], dim=0)
            else:
                inputs_embeds = torch.cat([bos_embeds, graph_embeds[i], text_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(device)
        attention_mask = torch.tensor(batch_attention_mask).to(device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                # do_sample=True,
                use_cache=True
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {'id': samples['id'],
                'pred': pred,
                'label': samples['label'],
                'desc': samples['desc'],
                'question': samples['question']}

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
