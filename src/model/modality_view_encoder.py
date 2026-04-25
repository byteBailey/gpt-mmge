"""
Modality-View Multimodal Graph Encoder (V) — 新方法主创新。

与现有 MultimodalGraphEncoder 平行存在, 设计要点:

  ① Modality-View Cross-Attention
     K queries 切成 K_text + K_image 两组:
       - text-view queries 只 attend to txt tokens
       - image-view queries 只 attend to img patches
     pool 后得到节点级独立的双流表示 (h_t, h_i).

  ② Edge-wise 4-Channel Modality Routing
     对每条边 (u → v) 学一个 4 维 softmax π_uv = (t→t, t→i, i→t, i→i),
     配合 4 个独立的投影矩阵 W_tt/W_ti/W_it/W_ii, 形成"模态消息" m_t / m_i;
     再做邻居层面 attention 聚合 (GAT-style q·k softmax).

  ③ Modality-Independent FFN
     text 和 image 分支各自走独立的 FFN + Norm, 不再共享.

输入/输出形状与 MultimodalGraphEncoder 完全兼容:
  forward(img_features, txt_features, neighbor_index, rel_ids, img_mask, txt_mask, neighbor_mask)
    → entity_repr [M, K_text + K_image, hidden_dim]
  其中前 K_text 个 token 是 text view, 后 K_image 个是 image view.
  外部 mean(dim=1) 仍可用 (与老 encoder 行为一致); 想保留模态区分可手动切片.

简化版未启用:
  - rel_ids 输入被忽略 (gpt 方案 6 的 pseudo-relation 暂未实现)
  - router 不接受 task query q (gpt 方案 4 的 task-conditioned 输入暂未实现)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Sub-module 1: Modality-View Cross-Attention
# ============================================================
class ModalityViewCrossAttention(nn.Module):
    """
    text-view K_t queries  → 只 attend to txt_features
    image-view K_i queries → 只 attend to img_features
    text 和 image 分支各自有独立的 q/k/v/out 投影 (彻底分离, 不共享参数).
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # text 分支
        self.q_proj_t = nn.Linear(dim, dim)
        self.k_proj_t = nn.Linear(dim, dim)
        self.v_proj_t = nn.Linear(dim, dim)
        self.out_proj_t = nn.Linear(dim, dim)
        self.norm_t = nn.LayerNorm(dim)

        # image 分支
        self.q_proj_i = nn.Linear(dim, dim)
        self.k_proj_i = nn.Linear(dim, dim)
        self.v_proj_i = nn.Linear(dim, dim)
        self.out_proj_i = nn.Linear(dim, dim)
        self.norm_i = nn.LayerNorm(dim)

        self.attn_drop = nn.Dropout(dropout)

    def _attend(self, q_proj, k_proj, v_proj, out_proj, queries, kv, kv_mask):
        """通用 multi-head cross-attn helper."""
        M, K, _ = queries.shape
        _, S, _ = kv.shape
        H, Dh = self.num_heads, self.head_dim
        Q = q_proj(queries).view(M, K, H, Dh).transpose(1, 2)   # [M, H, K, Dh]
        Kt = k_proj(kv).view(M, S, H, Dh).transpose(1, 2)       # [M, H, S, Dh]
        V = v_proj(kv).view(M, S, H, Dh).transpose(1, 2)        # [M, H, S, Dh]
        attn = (Q @ Kt.transpose(-2, -1)) * self.scale          # [M, H, K, S]
        if kv_mask is not None:
            attn = attn.masked_fill(~kv_mask[:, None, None, :].bool(), float('-inf'))
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        attn = attn.nan_to_num(0.0)
        out = (attn @ V).transpose(1, 2).reshape(M, K, -1)      # [M, K, D]
        return out_proj(out)

    def forward(
        self,
        entity_text: torch.Tensor,    # [M, K_t, D]
        entity_image: torch.Tensor,   # [M, K_i, D]
        img_feat: torch.Tensor,       # [M, P, D]
        txt_feat: torch.Tensor,       # [M, T, D]
        img_mask: torch.Tensor,       # [M, P]
        txt_mask: torch.Tensor,       # [M, T]
    ):
        out_t = self._attend(
            self.q_proj_t, self.k_proj_t, self.v_proj_t, self.out_proj_t,
            entity_text, txt_feat, txt_mask,
        )
        out_i = self._attend(
            self.q_proj_i, self.k_proj_i, self.v_proj_i, self.out_proj_i,
            entity_image, img_feat, img_mask,
        )
        entity_text = self.norm_t(entity_text + out_t)
        entity_image = self.norm_i(entity_image + out_i)
        return entity_text, entity_image


# ============================================================
# Sub-module 2: 4-Channel Modality Routing Graph Attention
# ============================================================
class ChannelRoutingGraphAttention(nn.Module):
    """
    边级 4 通道 modality routing + multi-head 邻居 attention.

    输入: 节点级 text/image 表示 h_t [M, D], h_i [M, D]
    输出: h_t_new [M, D], h_i_new [M, D]

    流程 (对每条边 u → v):
      ① π_uv = softmax( router_MLP([h_t_u; h_i_u; h_t_v; h_i_v]) ) ∈ R^4
         (4 通道分配在 head 间共享, 简化设计 — gpt advice2.txt 方案 C)
      ② 4 个独立投影 + 通道加权:
            m_t = π_uv[t→t] · W_tt(h_t_u) + π_uv[i→t] · W_it(h_i_u)
            m_i = π_uv[t→i] · W_ti(h_t_u) + π_uv[i→i] · W_ii(h_i_u)
      ③ 邻居层面 multi-head attention:
            q/k 投影后 reshape [..., H, Dh],
            每个 head 独立 softmax_u 在邻居维度上,
            α_uv^h ∈ [M, N, H]
      ④ 多头聚合 + concat + Add&Norm:
            head_h(t) = Σ_u α^h · m_t^h,
            h_t_new = LN(h_t + out_proj_t(concat_h head_h(t)))
    """

    def __init__(
        self, dim: int, num_heads: int = 8,
        router_hidden: int = 128, dropout: float = 0.1,
    ):
        super().__init__()
        assert dim % num_heads == 0, f'dim ({dim}) must be divisible by num_heads ({num_heads})'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 4 通道路由 MLP (head 间共享)
        self.router = nn.Sequential(
            nn.Linear(4 * dim, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, 4),
        )

        # 4 个独立投影 (modality channel projections, 投到完整 D 后 view 成 [H, Dh])
        self.W_tt = nn.Linear(dim, dim)  # text → text
        self.W_ti = nn.Linear(dim, dim)  # text → image
        self.W_it = nn.Linear(dim, dim)  # image → text
        self.W_ii = nn.Linear(dim, dim)  # image → image

        # 邻居层面 attention 投影 (双流独立, 多头)
        self.q_proj_t = nn.Linear(dim, dim)
        self.k_proj_t = nn.Linear(dim, dim)
        self.q_proj_i = nn.Linear(dim, dim)
        self.k_proj_i = nn.Linear(dim, dim)

        self.out_proj_t = nn.Linear(dim, dim)
        self.out_proj_i = nn.Linear(dim, dim)
        self.norm_t = nn.LayerNorm(dim)
        self.norm_i = nn.LayerNorm(dim)
        self.attn_drop = nn.Dropout(dropout)

        # ── 诊断 buffer: 累加 π 在有效边上的 running sum, 用于训练时观察 4 通道分布 ──
        # persistent=False → 不进 ckpt; reset_running_stats() 清零
        self.register_buffer('_pi_sum', torch.zeros(4), persistent=False)
        self.register_buffer('_pi_count', torch.zeros(1), persistent=False)

    @torch.no_grad()
    def get_pi_stats(self, reset: bool = True):
        """返回当前累计的 4 通道平均 [tt, ti, it, ii] (list of 4 floats), 或 None 若无有效边.
        reset=True 时同时清零累加器, 用于每 epoch 输出后重置."""
        if self._pi_count.item() < 1:
            return None
        avg = (self._pi_sum / self._pi_count).cpu().tolist()
        if reset:
            self._pi_sum.zero_()
            self._pi_count.zero_()
        return avg

    def _multihead_aggregate(self, h_v, m, q_proj, k_proj, out_proj, valid):
        """多头 attention 聚合: q from h_v [M, D], message m [M, N, D].
        返回 [M, D] (concat head 后再 out_proj).
        """
        M, N, D = m.shape
        H, Dh = self.num_heads, self.head_dim

        # q: [M, H, Dh]
        q = q_proj(h_v).view(M, H, Dh)
        # k: [M, N, H, Dh] → permute 到 [M, H, N, Dh]
        k = k_proj(m).view(M, N, H, Dh).permute(0, 2, 1, 3)
        # attention score: [M, H, N]
        attn = (q.unsqueeze(2) * k).sum(-1) * self.scale
        attn = attn.masked_fill(~valid.unsqueeze(1), float('-inf'))
        alpha = F.softmax(attn, dim=-1).nan_to_num(0.0)              # [M, H, N]
        alpha = self.attn_drop(alpha)

        # 聚合 message (按 head 重组): [M, H, N, Dh]
        m_h = m.view(M, N, H, Dh).permute(0, 2, 1, 3)
        head_out = (alpha.unsqueeze(-1) * m_h).sum(dim=2)            # [M, H, Dh]
        out = head_out.reshape(M, H * Dh)                            # [M, D]
        return out_proj(out)

    def forward(
        self,
        h_t: torch.Tensor,                 # [M, D]
        h_i: torch.Tensor,                 # [M, D]
        neighbor_index: torch.LongTensor,   # [M, N]
        neighbor_mask: torch.Tensor = None, # [M, N]
    ):
        M, N = neighbor_index.shape

        # 安全索引: padding 位置可能含非法索引, 先替换为 0
        if neighbor_mask is not None:
            valid = neighbor_mask.bool()
            safe_idx = neighbor_index.masked_fill(~valid, 0)
        else:
            valid = torch.ones(M, N, dtype=torch.bool, device=h_t.device)
            safe_idx = neighbor_index

        # 索引邻居 (从当前最新表示)
        h_t_n = h_t[safe_idx]                                       # [M, N, D]
        h_i_n = h_i[safe_idx]                                       # [M, N, D]
        # 中心节点广播
        h_t_v = h_t.unsqueeze(1).expand(-1, N, -1)                  # [M, N, D]
        h_i_v = h_i.unsqueeze(1).expand(-1, N, -1)

        # ── 4 通道路由 (head 间共享) ──
        router_in = torch.cat([h_t_n, h_i_n, h_t_v, h_i_v], dim=-1) # [M, N, 4D]
        route_logits = self.router(router_in)                        # [M, N, 4]
        route_logits = route_logits.masked_fill(~valid.unsqueeze(-1), float('-inf'))
        pi = F.softmax(route_logits, dim=-1).nan_to_num(0.0)         # [M, N, 4]

        # ── 诊断: 累加有效边上 π 的 running sum (训练时) ──
        if self.training:
            with torch.no_grad():
                valid_f = valid.float().unsqueeze(-1)                # [M, N, 1]
                self._pi_sum += (pi.detach().float() * valid_f).sum(dim=(0, 1))
                self._pi_count += valid_f.sum()

        pi_tt = pi[..., 0:1]                                          # [M, N, 1]
        pi_ti = pi[..., 1:2]
        pi_it = pi[..., 2:3]
        pi_ii = pi[..., 3:4]

        # ── 4 个独立投影 ──
        m_tt = self.W_tt(h_t_n)                                       # [M, N, D]
        m_ti = self.W_ti(h_t_n)
        m_it = self.W_it(h_i_n)
        m_ii = self.W_ii(h_i_n)

        # ── 通道加权得到 modality-specific 消息 ──
        m_t = pi_tt * m_tt + pi_it * m_it                             # [M, N, D]   流向 v 的 text
        m_i = pi_ti * m_ti + pi_ii * m_ii                             # [M, N, D]   流向 v 的 image

        # ── multi-head 邻居 attention 聚合 ──
        msg_t = self._multihead_aggregate(h_t, m_t, self.q_proj_t, self.k_proj_t,
                                          self.out_proj_t, valid)     # [M, D]
        msg_i = self._multihead_aggregate(h_i, m_i, self.q_proj_i, self.k_proj_i,
                                          self.out_proj_i, valid)

        h_t_new = self.norm_t(h_t + msg_t)
        h_i_new = self.norm_i(h_i + msg_i)
        return h_t_new, h_i_new


# ============================================================
# Layer: ModalityView CrossAttn → 4-Channel Routing → Modality FFN
# ============================================================
class ModalityViewGraphLayer(nn.Module):
    def __init__(
        self, dim: int, num_heads: int = 8,
        router_hidden: int = 128, ffn_mult: int = 4, dropout: float = 0.1,
    ):
        super().__init__()
        self.mm_attn = ModalityViewCrossAttention(dim, num_heads, dropout)
        self.routing = ChannelRoutingGraphAttention(dim, num_heads, router_hidden, dropout)
        # 模态独立 FFN
        self.ffn_t = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim * ffn_mult, dim), nn.Dropout(dropout),
        )
        self.ffn_i = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim * ffn_mult, dim), nn.Dropout(dropout),
        )
        self.ffn_norm_t = nn.LayerNorm(dim)
        self.ffn_norm_i = nn.LayerNorm(dim)

    def forward(
        self, entity_text, entity_image,
        img_feat, txt_feat, img_mask, txt_mask,
        neighbor_index, neighbor_mask,
    ):
        # ① Modality-view cross-attn
        entity_text, entity_image = self.mm_attn(
            entity_text, entity_image, img_feat, txt_feat, img_mask, txt_mask,
        )

        # ② 4-channel routing graph attn
        h_t = entity_text.mean(dim=1)                                 # [M, D]
        h_i = entity_image.mean(dim=1)
        h_t_new, h_i_new = self.routing(h_t, h_i, neighbor_index, neighbor_mask)
        # 增量回传到 K queries (保留 query 个体差异)
        delta_t = (h_t_new - h_t).unsqueeze(1)                        # [M, 1, D]
        delta_i = (h_i_new - h_i).unsqueeze(1)
        entity_text = entity_text + delta_t
        entity_image = entity_image + delta_i

        # ③ 模态独立 FFN + Add&Norm
        entity_text = self.ffn_norm_t(entity_text + self.ffn_t(entity_text))
        entity_image = self.ffn_norm_i(entity_image + self.ffn_i(entity_image))

        return entity_text, entity_image


# ============================================================
# Encoder: 完整对外接口 (forward 签名与 MultimodalGraphEncoder 一致)
# ============================================================
class ModalityViewGraphEncoder(nn.Module):
    """
    输入: 节点的 img patch features [M, P, img_dim], txt token features [M, T, txt_dim],
          子图邻接 (neighbor_index [M, N], neighbor_mask [M, N])
    输出: [M, K_text + K_image, hidden_dim]
          前 K_text 个是 text view, 后 K_image 个是 image view.
    """

    def __init__(
        self,
        img_dim: int = 1024,
        txt_dim: int = 768,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        num_heads: int = 8,
        K_text: int = 2,
        K_image: int = 2,
        router_hidden: int = 128,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        # 兼容现有调用签名 (这些参数在简化版被忽略)
        num_relations: int = 1,
        num_queries: int = None,
    ):
        super().__init__()
        self.K_text = K_text
        self.K_image = K_image
        self.hidden_dim = hidden_dim

        # 特征投影
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.txt_proj = nn.Linear(txt_dim, hidden_dim)

        # 双流 K queries
        self.text_queries = nn.Parameter(torch.randn(K_text, hidden_dim))
        self.image_queries = nn.Parameter(torch.randn(K_image, hidden_dim))
        nn.init.normal_(self.text_queries, std=0.02)
        nn.init.normal_(self.image_queries, std=0.02)

        # L 个 layer
        self.layers = nn.ModuleList([
            ModalityViewGraphLayer(hidden_dim, num_heads, router_hidden, ffn_mult, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm_t = nn.LayerNorm(hidden_dim)
        self.final_norm_i = nn.LayerNorm(hidden_dim)

    @property
    def num_queries(self):
        return self.K_text + self.K_image

    @property
    def entity_queries(self):
        # 兼容老代码 (Stage 1 query_diversity_loss 等可能引用) — 拼起来即可
        return torch.cat([self.text_queries, self.image_queries], dim=0)

    @torch.no_grad()
    def get_routing_stats(self, reset: bool = True):
        """收集每层 ChannelRoutingGraphAttention 的 π 4 通道平均.
        返回 list of [tt, ti, it, ii] (None 若该层暂无累计样本).
        """
        return [layer.routing.get_pi_stats(reset=reset) for layer in self.layers]

    def forward(
        self,
        img_features: torch.Tensor,         # [M, P, img_dim]
        txt_features: torch.Tensor,         # [M, T, txt_dim]
        neighbor_index: torch.LongTensor,    # [M, N]
        rel_ids: torch.LongTensor = None,    # [M, N]  简化版忽略
        img_mask: torch.Tensor = None,
        txt_mask: torch.Tensor = None,
        neighbor_mask: torch.Tensor = None,
        return_intermediate: bool = False,   # 兼容 deepstack 签名 (新 encoder 不支持, 只兜底)
    ):
        M = img_features.size(0)

        # 投影到 hidden_dim
        img_feat = self.img_proj(img_features)   # [M, P, D]
        txt_feat = self.txt_proj(txt_features)   # [M, T, D]

        # 默认 mask 全 1
        if img_mask is None:
            img_mask = torch.ones(M, img_feat.size(1), device=img_feat.device)
        if txt_mask is None:
            txt_mask = torch.ones(M, txt_feat.size(1), device=txt_feat.device)

        # 初始化 K queries: 共享 query + modality-specific pooled
        img_mask_f = img_mask.float()
        img_pooled = (img_feat * img_mask_f.unsqueeze(-1)).sum(dim=1) / \
                     img_mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)   # [M, D]
        txt_mask_f = txt_mask.float()
        txt_pooled = (txt_feat * txt_mask_f.unsqueeze(-1)).sum(dim=1) / \
                     txt_mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)

        entity_text = self.text_queries.unsqueeze(0) + txt_pooled.unsqueeze(1)    # [M, K_text, D]
        entity_image = self.image_queries.unsqueeze(0) + img_pooled.unsqueeze(1)  # [M, K_image, D]

        intermediates = [] if return_intermediate else None

        for layer in self.layers:
            entity_text, entity_image = layer(
                entity_text, entity_image, img_feat, txt_feat,
                img_mask, txt_mask, neighbor_index, neighbor_mask,
            )
            if return_intermediate:
                intermediates.append(torch.cat([entity_text, entity_image], dim=1))

        entity_text = self.final_norm_t(entity_text)
        entity_image = self.final_norm_i(entity_image)

        # 输出与 MultimodalGraphEncoder 一致: [M, K, D] (前 K_text 是 text view, 后 K_image 是 image view)
        out = torch.cat([entity_text, entity_image], dim=1)

        if return_intermediate:
            return out, intermediates
        return out
