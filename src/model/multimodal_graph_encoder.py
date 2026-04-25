"""
Multimodal Graph Encoder — 多模态图编码器 (K-query 版)
核心创新：交替堆叠 Multimodal Cross-Attention 和 Relational Graph Attention，
         使多模态融合与图结构传播在每一层相互增强。
         每个实体由 K 个 learnable query 表示 (类似 Q-Former)，
         cross-attention 拥有更强的信息提取能力。

输入：子图中 M 个实体的图片/文本特征 + 图结构 (neighbor_index, rel_ids)
输出：M × K 个 token [M, K, D]，每个实体的 K 个表示融合了多模态 + 邻域结构 + 关系语义

批处理：多个子图通过 collate_subgraphs() 合并为一个大图，
        neighbor_index 全局偏移，encoder 统一处理，输出按 batch 向量拆分。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Sub-module 1: Multimodal Cross-Attention (支持 K queries)
# ============================================================
class MultimodalCrossAttention(nn.Module):
    """
    K 个实体 query 同时关注自身的图片和文本特征 (Key/Value)。
    Q = W_q(entity_repr)        [M, K, D]
    K = W_k([img_patches; txt_tokens])  [M, S, D]
    V = W_v([img_patches; txt_tokens])  [M, S, D]
    output = softmax(Q·K^T / √d) · V   [M, K, D]
    K 个 query 独立地 attend 到同一组 S 个多模态 token。
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        entity_repr: torch.Tensor,    # [M, K, D]  K queries per entity
        mm_features: torch.Tensor,     # [M, S, D]  S = num_patches + text_len
        mm_mask: torch.Tensor = None,  # [M, S]     1=有效, 0=padding
    ) -> torch.Tensor:                 # [M, K, D]
        residual = entity_repr
        M, K, _ = entity_repr.shape
        S = mm_features.size(1)
        H, Dh = self.num_heads, self.head_dim

        Q = self.q_proj(entity_repr).view(M, K, H, Dh).transpose(1, 2)   # [M, H, K, Dh]
        Kt = self.k_proj(mm_features).view(M, S, H, Dh).transpose(1, 2)  # [M, H, S, Dh]
        V = self.v_proj(mm_features).view(M, S, H, Dh).transpose(1, 2)   # [M, H, S, Dh]

        # Scaled dot-product attention
        attn = (Q @ Kt.transpose(-2, -1)) * self.scale                   # [M, H, K, S]
        if mm_mask is not None:
            attn = attn.masked_fill(~mm_mask[:, None, None, :].bool(), float("-inf"))
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        attn = attn.nan_to_num(0.0)  # mask 全 0 时 softmax 产生 NaN → 置 0, 由残差连接保底
        out = (attn @ V).transpose(1, 2).reshape(M, K, -1)               # [M, K, D]
        out = self.out_proj(out)

        return self.norm(residual + out)  # Add & Norm, LayerNorm 作用于最后一维 D


# ============================================================
# Sub-module 2: Relational Graph Attention (不变，操作 [M, D])
# ============================================================
class RelationalGraphAttention(nn.Module):
    """
    关系感知图注意力：中心实体关注子图中的邻居实体。
    输入输出均为 [M, D]（由 Layer 负责 K queries 的 pool/broadcast）。
    Q = W_q(h_i)                    → 中心实体
    K = W_k(h_j + r_ij)             → 邻居 + 关系嵌入 (关系影响 Key)
    V = W_v(h_j)                    → 邻居 (Value 不含关系)
    α_j = softmax(Q·K_j^T / √d)    → 不同关系 → 不同 attention 权重
    output = Σ α_j · V_j
    """

    def __init__(
        self, dim: int, num_heads: int = 8,
        num_relations: int = 237, dropout: float = 0.1,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

        # 可训练的关系嵌入表
        self.rel_embeddings = nn.Embedding(num_relations, dim)

    def forward(
        self,
        entity_repr: torch.Tensor,          # [M, D]   子图中所有实体的表示
        neighbor_index: torch.LongTensor,    # [M, N]   每个实体的邻居在 entity_repr 中的索引
        rel_ids: torch.LongTensor,           # [M, N]   每条边的关系类型 ID
        neighbor_mask: torch.Tensor = None,  # [M, N]   1=有效邻居, 0=padding
    ) -> torch.Tensor:                       # [M, D]
        residual = entity_repr
        M = entity_repr.size(0)
        N = neighbor_index.size(1)
        H, Dh = self.num_heads, self.head_dim

        # 安全索引: padding 位置可能含非法索引, 先替换为 0
        safe_index = neighbor_index
        if neighbor_mask is not None:
            valid = neighbor_mask.bool()
            safe_index = neighbor_index.masked_fill(~valid, 0)
            # 检查有效邻居索引是否越界 (单节点退化子图时 valid 可能为空)
            valid_index = safe_index[valid]
            if valid_index.numel() > 0:
                assert valid_index.max() < M, \
                    f"neighbor_index out of range: max={valid_index.max().item()}, M={M}"
                assert valid_index.min() >= 0, \
                    f"neighbor_index has negative index: min={valid_index.min().item()}"

        # 从当前 entity_repr 动态获取邻居表示 (每层都是最新的)
        neighbor_repr = entity_repr[safe_index]                              # [M, N, D]

        # 安全关系 ID: padding 位置可能含非法 ID, 先替换为 0
        safe_rel_ids = rel_ids
        if neighbor_mask is not None:
            safe_rel_ids = rel_ids.masked_fill(~valid, 0)
            # 检查有效关系 ID 是否越界 (单节点退化子图时 valid 可能为空)
            valid_rel = safe_rel_ids[valid]
            if valid_rel.numel() > 0:
                num_rels = self.rel_embeddings.num_embeddings
                assert valid_rel.max() < num_rels, \
                    f"rel_ids out of range: max={valid_rel.max().item()}, num_relations={num_rels}"
                assert valid_rel.min() >= 0, \
                    f"rel_ids has negative id: min={valid_rel.min().item()}"

        # 关系嵌入
        rel_emb = self.rel_embeddings(safe_rel_ids)                          # [M, N, D]

        # Q 来自中心实体
        Q = self.q_proj(entity_repr).view(M, 1, H, Dh).transpose(1, 2)      # [M, H, 1, Dh]
        # K = W_k(neighbor + relation)  → 关系加入 Key 端
        K = self.k_proj(neighbor_repr + rel_emb).view(M, N, H, Dh).transpose(1, 2)  # [M, H, N, Dh]
        # V = W_v(neighbor)             → Value 不含关系
        V = self.v_proj(neighbor_repr).view(M, N, H, Dh).transpose(1, 2)            # [M, H, N, Dh]

        # Attention: 关系嵌入使同一邻居在不同关系下产生不同权重
        attn = (Q @ K.transpose(-2, -1)) * self.scale                       # [M, H, 1, N]
        if neighbor_mask is not None:
            attn = attn.masked_fill(~neighbor_mask[:, None, None, :].bool(), float("-inf"))
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        attn = attn.nan_to_num(0.0)  # 孤立节点 (无邻居) → 0 权重, 由残差连接保底
        out = (attn @ V).transpose(1, 2).reshape(M, -1)                     # [M, D]
        out = self.out_proj(out)

        return self.norm(residual + out)  # Add & Norm


# ============================================================
# 交替层: MultimodalCrossAttn → RelationalGraphAttn → FFN
# ============================================================
class MultimodalGraphLayer(nn.Module):
    """
    一个交替层，包含三个子模块：
      ① Multimodal Cross-Attention — K queries 关注实体自身的多模态特征 [M, K, D]
      ② Relational Graph Attention  — 对 K queries 做 mean pool → [M, D],
         图注意力聚合邻居信息 → 将图更新广播回 K queries
      ③ FFN — 逐 query 非线性变换 + 残差
    """

    def __init__(
        self, dim: int, num_heads: int = 8,
        num_relations: int = 237, ffn_mult: int = 4, dropout: float = 0.1,
    ):
        super().__init__()
        # ① 多模态融合 (K queries)
        self.mm_attn = MultimodalCrossAttention(dim, num_heads, dropout)
        # ② 图传播 (pooled [M, D])
        self.graph_attn = RelationalGraphAttention(dim, num_heads, num_relations, dropout)
        # ③ FFN (逐 query)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(
        self,
        entity_repr,      # [M, K, D]
        mm_features,       # [M, S, D]
        mm_mask,           # [M, S]
        neighbor_index,    # [M, N]
        rel_ids,           # [M, N]
        neighbor_mask,     # [M, N]
    ):
        # ① Multimodal Cross-Attention: K queries 关注自己的图片/文本
        entity_repr = self.mm_attn(entity_repr, mm_features, mm_mask)  # [M, K, D]

        # ② Relational Graph Attention: pool → 图注意力 → broadcast
        #    mean pool K queries → 每个实体一个向量, 用于图传播
        pooled = entity_repr.mean(dim=1)                                # [M, D]
        graph_out = self.graph_attn(pooled, neighbor_index, rel_ids, neighbor_mask)  # [M, D]
        #    将图注意力的增量广播回每个 query:
        #    entity_repr[m,k] = entity_repr[m,k] + (graph_out[m] - pooled[m])
        #    保留各 query 的个体差异, 同时融入邻居结构信息
        entity_repr = entity_repr + (graph_out - pooled).unsqueeze(1)  # [M, K, D]

        # ③ FFN + Add & Norm (逐 query, nn.Linear 自动处理 [M, K, D])
        entity_repr = self.ffn_norm(entity_repr + self.ffn(entity_repr))

        return entity_repr  # [M, K, D]


# ============================================================
# 完整编码器: 特征投影 → 交替层 × L → 输出 M×K 个 token
# ============================================================
class MultimodalGraphEncoder(nn.Module):
    """
    多模态图编码器 (K-query 版)。
    输入：子图中 M 个实体的图片特征(CLIP)、文本特征(TextEncoder)、图结构
    输出：[M, K, D]，每个实体的 K 个 token，融合了多模态 + 邻域结构 + 关系语义

    Pipeline:
      img_features ─→ img_proj ───┐
      txt_features ─→ txt_proj ───┘──→ mm_features [M, S, D]
      K entity_queries + pooled_mm  ─→ entity_repr [M, K, D]
                                   ─→ [MultimodalGraphLayer × L] ─→ [M, K, D]

    批处理: 多个子图通过 collate_subgraphs() 合并为一个大图,
            neighbor_index 全局偏移, encoder 统一处理。
    """

    def __init__(
        self,
        img_dim: int = 1024,       # CLIP ViT-L patch feature dim
        txt_dim: int = 768,        # Text encoder hidden dim
        hidden_dim: int = 512,     # 编码器统一隐藏维度
        num_layers: int = 2,       # 交替层数量
        num_heads: int = 8,        # 注意力头数
        num_relations: int = 237,  # 关系类型数量 (DB15K=237)
        num_queries: int = 1,      # 每个实体的 learnable query 数量 (K)
        ffn_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries

        # 特征投影: 将不同维度的输入统一到 hidden_dim
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.txt_proj = nn.Linear(txt_dim, hidden_dim)

        # K 个可学习的实体 query (与 mm_mean 叠加产生个体差异)
        self.entity_queries = nn.Parameter(torch.randn(num_queries, hidden_dim))
        nn.init.normal_(self.entity_queries, std=0.02)

        # L 个交替层
        self.layers = nn.ModuleList([
            MultimodalGraphLayer(hidden_dim, num_heads, num_relations, ffn_mult, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        img_features: torch.Tensor,         # [M, P, img_dim]  每个实体的 CLIP patch features
        txt_features: torch.Tensor,          # [M, T, txt_dim]  每个实体的 text token features
        neighbor_index: torch.LongTensor,    # [M, N]  每个实体的邻居索引 (指向 0..M-1)
        rel_ids: torch.LongTensor,           # [M, N]  每条边的关系类型 ID
        img_mask: torch.Tensor = None,       # [M, P]
        txt_mask: torch.Tensor = None,       # [M, T]
        neighbor_mask: torch.Tensor = None,  # [M, N]
    ) -> torch.Tensor:                       # [M, K, hidden_dim]
        M = img_features.size(0)
        K = self.num_queries

        # ── 特征投影到统一维度 ──
        img_feat = self.img_proj(img_features)   # [M, P, D]
        txt_feat = self.txt_proj(txt_features)   # [M, T, D]

        # ── 拼接多模态特征 ──
        mm_features = torch.cat([img_feat, txt_feat], dim=1)  # [M, P+T, D]

        # ── 构建多模态 mask (独立处理, 缺失的默认全 1) ──
        if img_mask is None:
            img_mask = torch.ones(M, img_feat.size(1), device=img_feat.device)
        if txt_mask is None:
            txt_mask = torch.ones(M, txt_feat.size(1), device=txt_feat.device)
        mm_mask = torch.cat([img_mask, txt_mask], dim=1)  # [M, P+T]

        # ── 初始化 K queries: 共享 queries + 各实体的 mm_mean, 每个实体有 K 个不同的初始表示 ──
        mm_mask_f = mm_mask.float()                                        # [M, S]
        mm_sum = (mm_features * mm_mask_f.unsqueeze(-1)).sum(dim=1)        # [M, D]
        mm_den = mm_mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)        # [M, 1]
        pooled_mm = mm_sum / mm_den                                        # [M, D]
        # [1, K, D] + [M, 1, D] → [M, K, D]
        entity_repr = self.entity_queries.unsqueeze(0) + pooled_mm.unsqueeze(1)

        # ── 通过 L 个交替层 ──
        for layer in self.layers:
            entity_repr = layer(
                entity_repr,      # [M, K, D] — K queries per entity
                mm_features,      # [M, S, D] — 每个实体的多模态特征 (每层都重新关注)
                mm_mask,
                neighbor_index,   # 图结构: 邻居索引 (指向子图内实体)
                rel_ids,          # 边的关系类型
                neighbor_mask,
            )

        return self.final_norm(entity_repr)  # [M, K, D]


# ============================================================
# 投影层: 编码器输出 → LLM 嵌入空间
# ============================================================
class Projector(nn.Module):
    """
    将编码器输出投影到 LLM 的 token embedding 空间。
    支持任意 batch 维度: [M, D] → [M, llm_dim] 或 [M, K, D] → [M, K, llm_dim]。
    """

    def __init__(self, encoder_dim: int = 512, llm_dim=None):
        super().__init__()
        if llm_dim is None:
            raise ValueError('llm_dim must be provided explicitly.')
        self.proj = nn.Sequential(
            nn.Linear(encoder_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, entity_repr: torch.Tensor) -> torch.Tensor:
        return self.proj(entity_repr)  # [..., llm_dim]


# ============================================================
# 工具函数: 多子图批处理
# ============================================================
def collate_subgraphs(subgraphs: list) -> tuple:
    """
    将多个子图合并为一个大图, 用于批处理。
    每个子图的 neighbor_index 自动加上全局偏移量,
    使索引从局部 (0..M_i-1) 变为全局 (0..M_total-1)。
    Args:
        subgraphs: list of dict, 每个 dict 包含:
            - img_features:  [M_i, P, img_dim]
            - txt_features:  [M_i, T, txt_dim]
            - neighbor_index: [M_i, N]
            - rel_ids:        [M_i, N]
            - neighbor_mask:  [M_i, N]
            - img_mask (可选): [M_i, P]
            - txt_mask (可选): [M_i, T]

    Returns:
        batched: dict, 合并后的输入 (直接传给 encoder.forward)
        batch_vec: [M_total] 每个实体属于哪个子图 (0, 1, 2, ...)
    """
    offset = 0
    all_img, all_txt = [], []
    all_neighbor_index, all_rel_ids, all_neighbor_mask = [], [], []
    all_img_mask, all_txt_mask = [], []
    batch_ids = []

    for i, sg in enumerate(subgraphs):
        M_i = sg["img_features"].size(0)
        all_img.append(sg["img_features"])
        all_txt.append(sg["txt_features"])
        # neighbor_index: 只偏移有效位置, padding 位保持 0
        ni = sg["neighbor_index"].clone()
        mask_bool = sg["neighbor_mask"].bool()
        ni[mask_bool] += offset
        all_neighbor_index.append(ni)
        all_rel_ids.append(sg["rel_ids"])
        all_neighbor_mask.append(sg["neighbor_mask"])
        # mask: 有则用, 无则补全 1 (统一处理, 避免部分子图有 mask 部分没有的不一致)
        if "img_mask" in sg:
            all_img_mask.append(sg["img_mask"])
        else:
            all_img_mask.append(torch.ones(M_i, sg["img_features"].size(1),
                                           device=sg["img_features"].device))
        if "txt_mask" in sg:
            all_txt_mask.append(sg["txt_mask"])
        else:
            all_txt_mask.append(torch.ones(M_i, sg["txt_features"].size(1),
                                           device=sg["txt_features"].device))
        batch_ids.extend([i] * M_i)
        offset += M_i

    batched = {
        "img_features": torch.cat(all_img, dim=0),
        "txt_features": torch.cat(all_txt, dim=0),
        "neighbor_index": torch.cat(all_neighbor_index, dim=0),
        "rel_ids": torch.cat(all_rel_ids, dim=0),
        "neighbor_mask": torch.cat(all_neighbor_mask, dim=0),
        "img_mask": torch.cat(all_img_mask, dim=0),
        "txt_mask": torch.cat(all_txt_mask, dim=0),
    }

    device = batched["img_features"].device
    batch_vec = torch.tensor(batch_ids, dtype=torch.long, device=device)
    return batched, batch_vec
