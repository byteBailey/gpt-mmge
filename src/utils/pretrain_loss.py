"""
Stage 1 预训练损失:
- 对称 InfoNCE (SimCLR 风格): 跨数据集 in-batch 负样本
- K-query 多样性正则: 防 K 个 learnable query 坍塌成同一向量
"""

import torch
import torch.nn.functional as F


def info_nce_symmetric(z, batch_size, tau=0.1):
    """
    对称 InfoNCE 损失。
    z: [2B, D] L2-normalized, 前 B 个是 anchor, 后 B 个是 positive。
    每个 anchor i 与 positive i (位置 i+B) 互为正对, 其余 2B-2 个为负样本。
    """
    B = batch_size
    N = z.size(0)
    assert N == 2 * B

    sim = z @ z.t() / tau                                  # [2B, 2B]
    sim.masked_fill_(torch.eye(N, dtype=torch.bool, device=z.device), float('-inf'))

    pos_idx = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B, device=z.device),
    ])                                                      # [2B]

    log_prob = F.log_softmax(sim, dim=-1)
    loss = -log_prob[torch.arange(N, device=z.device), pos_idx].mean()
    return loss


def query_diversity_loss(queries):
    """
    K-query 多样性正则: 鼓励 K 个 learnable query 在余弦空间互相不重合。
    queries: [K, D] (encoder.entity_queries)
    返回 off-diagonal cosine similarity 的平均平方。
    """
    K = queries.size(0)
    if K == 1:
        return torch.zeros((), device=queries.device, dtype=queries.dtype)
    q = F.normalize(queries, dim=-1)
    sim = q @ q.t()                                         # [K, K]
    eye = torch.eye(K, device=q.device, dtype=q.dtype)
    off = sim - eye
    return (off ** 2).sum() / (K * (K - 1))


def cross_modal_alignment_loss(z_t, z_i, tau=0.1):
    """
    CLIP 风格同节点跨模态对齐 loss.
    z_t, z_i: [N, D] 同一组节点的 text-view 和 image-view 表示, L2-normalized.
    第 k 行的 z_t[k] 与 z_i[k] 应是同节点; 跨节点为负.
    返回对称 (text→image + image→text) InfoNCE.
    """
    N = z_t.size(0)
    sim = z_t @ z_i.t() / tau                                # [N, N]
    target = torch.arange(N, device=z_t.device)
    loss_t2i = F.cross_entropy(sim, target)
    loss_i2t = F.cross_entropy(sim.t(), target)
    return 0.5 * (loss_t2i + loss_i2t)
