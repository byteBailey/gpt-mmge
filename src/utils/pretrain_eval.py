"""
Stage 1 预训练评测:
- k-NN @k 分类 (在 train embeddings 上找最近邻)
- alignment / uniformity (Wang & Isola 2020)
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.pretrain_dataset import MMGEvalDataset, eval_collate


@torch.no_grad()
def encode_nodes(encoder, graph, node_ids, builder, device,
                 batch_size=64, num_workers=0, desc=None, amp_dtype=None):
    """对给定节点列表逐 batch 提取 [N, D] 表征 (K-query mean-pool, L2 norm)."""
    encoder.eval()
    ds = MMGEvalDataset(graph, node_ids, builder)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=eval_collate,
                        pin_memory=(num_workers > 0))

    feats, labels = [], []
    iterable = loader if desc is None else tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for batch in iterable:
        batched = {k: v.to(device, non_blocking=True) for k, v in batch['batched'].items()}
        if amp_dtype is not None:
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                repr = encoder(**batched)
        else:
            repr = encoder(**batched)
        center = repr[batch['center_indices'].to(device, non_blocking=True)]
        z = center.mean(dim=1)
        z = F.normalize(z.float(), dim=-1)                  # 归一化前转 fp32 避免精度损失
        feats.append(z.cpu())
        labels.append(batch['labels'])
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def knn_accuracy(train_z, train_y, val_z, val_y, k=5):
    """k-NN 多数投票分类准确率 (cosine 相似度, 表征已 L2 归一化)。"""
    sim = val_z @ train_z.t()                              # [N_val, N_train]
    topk_idx = sim.topk(k, dim=-1).indices                  # [N_val, k]
    topk_labels = train_y[topk_idx]                         # [N_val, k]
    pred = topk_labels.mode(dim=-1).values                  # [N_val]
    return (pred == val_y).float().mean().item()


def alignment_uniformity(z_anchor, z_pos):
    """
    Wang & Isola (2020) 的两个 SSL 健康度指标。
    z_anchor, z_pos: [B, D], L2-normalized.
    返回 (alignment, uniformity).
      alignment: E[||z_a - z_p||^2], 越小 → 正样本对越近.
      uniformity: log E[exp(-2 * ||z_i - z_j||^2)], 越小 → 在球面越均匀.
    """
    align = ((z_anchor - z_pos).pow(2).sum(dim=-1)).mean().item()

    z = torch.cat([z_anchor, z_pos], dim=0)                 # [2B, D]
    pdist_sq = torch.cdist(z, z, p=2).pow(2)                # [2B, 2B]
    N = z.size(0)
    mask = ~torch.eye(N, dtype=torch.bool, device=z.device)
    unif = (-2 * pdist_sq[mask]).exp().mean().log().item()
    return align, unif


@torch.no_grad()
def evaluate_knn(encoder, dataset_names, graphs, splits, builders, device,
                 k=5, batch_size=64, num_workers=0, val_subsample=None,
                 train_subsample=None, amp_dtype=None):
    """对每个数据集计算 val 上的 k-NN@k 准确率。

    val_subsample / train_subsample: 限制 val / train 节点数 (加速 eval).
        train 集仅作 k-NN 检索库, 数千个候选已足够稳定;
        val 集决定 acc 估计精度, 1000 时 95% CI ≈ ±3%.
    amp_dtype: 若指定 (bf16/fp16), eval forward 也走 AMP, 加快推理.
    """
    encoder.eval()
    results = {}
    pbar = tqdm(list(zip(dataset_names, graphs, splits, builders)),
                desc='[eval] knn', leave=False, dynamic_ncols=True)
    for name, graph, split, builder in pbar:
        train_ids = split['train']
        val_ids = split['val']
        if val_subsample is not None and len(val_ids) > val_subsample:
            val_ids = val_ids[:val_subsample]
        if train_subsample is not None and len(train_ids) > train_subsample:
            train_ids = train_ids[:train_subsample]

        train_z, train_y = encode_nodes(encoder, graph, train_ids, builder,
                                        device, batch_size, num_workers,
                                        desc=f'  {name} train enc', amp_dtype=amp_dtype)
        val_z, val_y = encode_nodes(encoder, graph, val_ids, builder,
                                    device, batch_size, num_workers,
                                    desc=f'  {name} val enc', amp_dtype=amp_dtype)
        acc = knn_accuracy(train_z, train_y, val_z, val_y, k=k)
        results[name] = acc
        pbar.set_postfix({name: f'{acc:.3f}'})
    pbar.close()
    results['_avg'] = sum(v for k_, v in results.items() if k_ != '_avg') / len(dataset_names)
    return results
