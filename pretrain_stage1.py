"""
Stage 1 多模态图编码器预训练。

对齐 MLaGA 的"structure-aware multimodal aligner"思路:
  - 跨多个 MAGB 数据集 (Movies/Toys/Grocery/Arts/CD) 联合训练
  - 主 loss: 结构对比 InfoNCE (anchor + 1-hop 邻居 positive, in-batch negatives)
  - 辅 loss: K-query 多样性正则
  - 评测: 每 epoch 在 val 上跑 k-NN@5 + alignment/uniformity

输出 ckpt 仅包含 encoder + projection head (无 LLM), 后续 Stage 2 用作 frozen / 初始化。
"""

import argparse
import json
import logging
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.seed import seed_everything
from src.dataset.pretrain_dataset import (
    MMGPretrainDataset, pretrain_collate, load_graphs_and_splits,
)
from src.model.multimodal_graph_encoder import MultimodalGraphEncoder
from src.utils.pretrain_loss import (
    info_nce_symmetric, query_diversity_loss, cross_modal_alignment_loss,
)
from src.utils.pretrain_eval import evaluate_knn, alignment_uniformity


def parse_args():
    p = argparse.ArgumentParser('stage1_pretrain')
    p.add_argument('--datasets', nargs='+',
                   default=['movies', 'toys', 'grocery', 'arts', 'cd'])
    p.add_argument('--output_dir', type=str, default='output_stage1/exp1')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str, default='cuda:0')

    # 训练
    p.add_argument('--num_epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--wd', type=float, default=0.05)
    p.add_argument('--warmup_epochs', type=float, default=1.0)
    p.add_argument('--min_lr', type=float, default=1e-6)
    p.add_argument('--clip_grad', type=float, default=1.0)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--amp', type=str, default='none', choices=['none', 'bf16', 'fp16'],
                   help='混合精度训练: bf16 (Ampere+) / fp16 / none')

    # Loss
    p.add_argument('--tau', type=float, default=0.1, help='InfoNCE 温度')
    p.add_argument('--div_lambda', type=float, default=0.01, help='K-query 多样性权重')
    # view-aware loss (仅 encoder_type=view 生效, mmge 路径下 lambda_t/i/align 自动忽略)
    p.add_argument('--lambda_t', type=float, default=0.5,
                   help='text-view 单独结构对比 InfoNCE 权重')
    p.add_argument('--lambda_i', type=float, default=0.5,
                   help='image-view 单独结构对比 InfoNCE 权重')
    p.add_argument('--lambda_align', type=float, default=1.0,
                   help='同节点 text-image 跨模态对齐 (CLIP 风格) 权重')

    # 子图
    p.add_argument('--num_hops', type=int, default=2)
    p.add_argument('--max_neighbors', type=int, default=20)

    # MMGE
    p.add_argument('--mm_num_layers', type=int, default=4)
    p.add_argument('--mm_num_heads', type=int, default=8)
    p.add_argument('--mm_hidden_dim', type=int, default=1024)
    p.add_argument('--num_queries', type=int, default=4)
    p.add_argument('--proj_dim', type=int, default=256, help='对比头投影维度')
    p.add_argument('--dropout', type=float, default=0.1)

    # 评测
    p.add_argument('--eval_every', type=int, default=1)
    p.add_argument('--eval_k', type=int, default=5)
    p.add_argument('--val_subsample', type=int, default=1000,
                   help='每个数据集 val 评测节点数上限 (加速)')
    p.add_argument('--train_subsample', type=int, default=3000,
                   help='每个数据集 k-NN 检索库 (train 端) 节点数上限 (加速 eval)')
    p.add_argument('--eval_batch_size', type=int, default=128,
                   help='eval 时的 batch size, 没有 backward 可以更大')

    # Encoder type
    p.add_argument('--encoder_type', type=str, default='mmge', choices=['mmge', 'view'],
                   help="'mmge': 老 K-query 单流 encoder; "
                        "'view': Modality-view + 4-channel routing 双流 encoder")
    p.add_argument('--K_text', type=int, default=2)
    p.add_argument('--K_image', type=int, default=2)
    p.add_argument('--router_hidden', type=int, default=128)

    return p.parse_args()


def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'pretrain.log')
    logger = logging.getLogger('stage1')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info(f'log file: {log_path}')
    return logger


class Stage1Model(nn.Module):
    """MMGE / view-encoder + 对比投影头.

    encoder_type=view 时, encoder 输出 [M, K_text + K_image, D],
    Stage 1 同时计算:
      - z_all: K 维 mean pool 后的整体表示 (向后兼容老 loss)
      - z_t:   前 K_text 个 query 的 mean → text-view 表示
      - z_i:   后 K_image 个 query 的 mean → image-view 表示
    投影头由 z_t / z_i / z_all 共享 (单一 proj_head).
    """

    def __init__(self, args, sample_graph):
        super().__init__()
        img_dim = sample_graph.img_features.shape[-1]
        txt_dim = sample_graph.txt_features.shape[-1]

        self.encoder_type = args.encoder_type

        if args.encoder_type == 'view':
            from src.model.modality_view_encoder import ModalityViewGraphEncoder
            print(f'[stage1] using ModalityViewGraphEncoder '
                  f'(K_text={args.K_text}, K_image={args.K_image}, '
                  f'router_hidden={args.router_hidden})')
            self.encoder = ModalityViewGraphEncoder(
                img_dim=img_dim,
                txt_dim=txt_dim,
                hidden_dim=args.mm_hidden_dim,
                num_layers=args.mm_num_layers,
                num_heads=args.mm_num_heads,
                K_text=args.K_text,
                K_image=args.K_image,
                router_hidden=args.router_hidden,
                dropout=args.dropout,
            )
            self.K_text = args.K_text
            self.K_image = args.K_image
            self.num_queries = args.K_text + args.K_image
        else:
            print(f'[stage1] using legacy MultimodalGraphEncoder '
                  f'(num_queries={args.num_queries})')
            self.encoder = MultimodalGraphEncoder(
                img_dim=img_dim,
                txt_dim=txt_dim,
                hidden_dim=args.mm_hidden_dim,
                num_layers=args.mm_num_layers,
                num_heads=args.mm_num_heads,
                num_relations=1,
                num_queries=args.num_queries,
                dropout=args.dropout,
            )
            self.K_text = self.K_image = None
            self.num_queries = args.num_queries

        self.proj_head = nn.Sequential(
            nn.Linear(args.mm_hidden_dim, args.mm_hidden_dim),
            nn.GELU(),
            nn.Linear(args.mm_hidden_dim, args.proj_dim),
        )

    def _proj(self, x):
        return F.normalize(self.proj_head(x), dim=-1)

    def forward(self, batched, center_indices):
        """
        Returns dict:
          - 'z_all': [2B, proj_dim], 整体表示 (mean K)
          - 'z_t':   [2B, proj_dim], text-view 表示 (仅 view encoder, mmge 时为 None)
          - 'z_i':   [2B, proj_dim], image-view 表示 (仅 view encoder)
        """
        repr = self.encoder(**batched)                      # [M, K, D]
        center = repr[center_indices]                        # [2B, K, D]

        z_all = self._proj(center.mean(dim=1))               # [2B, proj_dim]
        if self.encoder_type == 'view':
            center_t = center[:, :self.K_text].mean(dim=1)   # [2B, D]
            center_i = center[:, self.K_text:].mean(dim=1)   # [2B, D]
            z_t = self._proj(center_t)
            z_i = self._proj(center_i)
            return {'z_all': z_all, 'z_t': z_t, 'z_i': z_i}
        return {'z_all': z_all, 'z_t': None, 'z_i': None}


def adjust_lr(optimizer, epoch_float, args):
    """cosine LR schedule with warmup."""
    if epoch_float < args.warmup_epochs:
        lr = args.lr * epoch_float / args.warmup_epochs
    else:
        progress = (epoch_float - args.warmup_epochs) / max(1, args.num_epochs - args.warmup_epochs)
        import math
        lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


def save_ckpt(model, optimizer, epoch, args, tag='best'):
    path = os.path.join(args.output_dir, f'stage1_{tag}.pt')
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'args': vars(args),
    }, path)
    return path


def main():
    args = parse_args()
    seed_everything(args.seed)
    logger = setup_logger(args.output_dir)
    logger.info(f'args: {json.dumps(vars(args), indent=2)}')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ── 数据 ──
    graphs, splits = load_graphs_and_splits(args.datasets)
    train_ds = MMGPretrainDataset(graphs, splits,
                                  num_hops=args.num_hops,
                                  max_neighbors=args.max_neighbors)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True,
                              num_workers=args.num_workers,
                              collate_fn=pretrain_collate,
                              pin_memory=(args.num_workers > 0),
                              persistent_workers=(args.num_workers > 0))
    builders = train_ds.builders

    # ── 模型 ──
    model = Stage1Model(args, graphs[0]).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    logger.info(f'model trainable: {n_train:,} / {n_all:,} ({100*n_train/n_all:.2f}%)')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.wd, betas=(0.9, 0.95))

    # AMP 上下文 (bf16 在 Ampere+ 无需 GradScaler)
    amp_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16}.get(args.amp)
    use_amp = amp_dtype is not None
    scaler = torch.cuda.amp.GradScaler() if args.amp == 'fp16' else None
    logger.info(f'AMP: {args.amp} (use_amp={use_amp}, scaler={scaler is not None})')

    best_knn = -1.0
    best_epoch = -1
    steps_per_epoch = len(train_loader)
    logger.info(f'steps per epoch: {steps_per_epoch}, total epochs: {args.num_epochs}')

    def _compute_losses(out, B):
        """统一的 loss 组装. out = {z_all, z_t, z_i}.
        view encoder: 主结构 InfoNCE(z_all) + λ_t·InfoNCE(z_t) + λ_i·InfoNCE(z_i) + λ_align·CLIP(z_t,z_i) + div
        mmge: 仅 InfoNCE(z_all) + div
        返回: (total_loss, dict 各项 scalar)
        """
        z_all = out['z_all']
        loss_nce = info_nce_symmetric(z_all, B, tau=args.tau)
        loss_div = query_diversity_loss(model.encoder.entity_queries)
        total = loss_nce + args.div_lambda * loss_div
        parts = {'nce': loss_nce, 'div': loss_div, 'nce_t': None, 'nce_i': None, 'align': None}

        if out['z_t'] is not None and args.encoder_type == 'view':
            z_t, z_i = out['z_t'], out['z_i']
            loss_nce_t = info_nce_symmetric(z_t, B, tau=args.tau)
            loss_nce_i = info_nce_symmetric(z_i, B, tau=args.tau)
            loss_align = cross_modal_alignment_loss(z_t, z_i, tau=args.tau)
            total = total + args.lambda_t * loss_nce_t \
                          + args.lambda_i * loss_nce_i \
                          + args.lambda_align * loss_align
            parts.update({'nce_t': loss_nce_t, 'nce_i': loss_nce_i, 'align': loss_align})
        return total, parts

    for epoch in range(args.num_epochs):
        model.train()
        epoch_acc = {k: 0.0 for k in ['loss', 'nce', 'div', 'nce_t', 'nce_i', 'align', 'au_align', 'au_unif']}
        t0 = time.time()

        pbar = tqdm(train_loader, total=steps_per_epoch,
                    desc=f'epoch {epoch}/{args.num_epochs}', dynamic_ncols=True)
        for step, batch in enumerate(pbar):
            lr = adjust_lr(optimizer, epoch + step / steps_per_epoch, args)
            optimizer.zero_grad()

            batched = {k: v.to(device, non_blocking=True) for k, v in batch['batched'].items()}
            center_indices = batch['center_indices'].to(device, non_blocking=True)
            B = batch['batch_size']

            if use_amp:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    out = model(batched, center_indices)
                    loss, parts = _compute_losses(out, B)
            else:
                out = model(batched, center_indices)
                loss, parts = _compute_losses(out, B)

            if scaler is not None:
                scaler.scale(loss).backward()
                if args.clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()

            with torch.no_grad():
                z_all = out['z_all']
                au_a, au_u = alignment_uniformity(z_all[:B].detach(), z_all[B:].detach())

            epoch_acc['loss']  += loss.item()
            epoch_acc['nce']   += parts['nce'].item()
            epoch_acc['div']   += parts['div'].item()
            for k in ('nce_t', 'nce_i', 'align'):
                if parts[k] is not None:
                    epoch_acc[k] += parts[k].item()
            epoch_acc['au_align'] += au_a
            epoch_acc['au_unif']  += au_u

            pbar.set_postfix({
                'lr': f'{lr:.2e}',
                'loss': f'{loss.item():.3f}',
                'nce': f'{parts["nce"].item():.2f}',
                **({'nce_t': f'{parts["nce_t"].item():.2f}',
                    'nce_i': f'{parts["nce_i"].item():.2f}',
                    'al': f'{parts["align"].item():.2f}'} if parts['nce_t'] is not None else {}),
                'au_a': f'{au_a:.2f}',
            })

            if (step + 1) % 200 == 0:
                msg = (f'epoch {epoch} step {step+1}/{steps_per_epoch} lr={lr:.2e} '
                       f'loss={loss.item():.4f} nce={parts["nce"].item():.4f} '
                       f'div={parts["div"].item():.4f}')
                if parts['nce_t'] is not None:
                    msg += (f' nce_t={parts["nce_t"].item():.4f} '
                            f'nce_i={parts["nce_i"].item():.4f} '
                            f'align={parts["align"].item():.4f}')
                msg += f' au_align={au_a:.3f} au_unif={au_u:.3f}'
                logger.info(msg)
        pbar.close()

        n = max(1, steps_per_epoch)
        msg = (f'epoch {epoch} done in {time.time()-t0:.1f}s | '
               f'avg_loss={epoch_acc["loss"]/n:.4f} nce={epoch_acc["nce"]/n:.4f} '
               f'div={epoch_acc["div"]/n:.4f}')
        if args.encoder_type == 'view':
            msg += (f' nce_t={epoch_acc["nce_t"]/n:.4f} '
                    f'nce_i={epoch_acc["nce_i"]/n:.4f} '
                    f'align={epoch_acc["align"]/n:.4f}')
        msg += (f' au_align={epoch_acc["au_align"]/n:.3f} '
                f'au_unif={epoch_acc["au_unif"]/n:.3f}')
        logger.info(msg)

        # ── 4 通道 π 诊断: 输出每层 routing 的平均 [tt, ti, it, ii] ──
        if hasattr(model.encoder, 'get_routing_stats'):
            stats = model.encoder.get_routing_stats(reset=True)
            for li, s in enumerate(stats):
                if s is None:
                    continue
                logger.info(f'  routing π L{li}: '
                            f'tt={s[0]:.3f} ti={s[1]:.3f} '
                            f'it={s[2]:.3f} ii={s[3]:.3f} '
                            f'(diag={s[0]+s[3]:.2f}, cross={s[1]+s[2]:.2f})')

        if (epoch + 1) % args.eval_every == 0:
            t1 = time.time()
            knn = evaluate_knn(model.encoder, args.datasets, graphs, splits,
                               builders, device, k=args.eval_k,
                               batch_size=args.eval_batch_size,
                               num_workers=args.num_workers,
                               val_subsample=args.val_subsample,
                               train_subsample=args.train_subsample,
                               amp_dtype=amp_dtype)
            knn_str = ' '.join(f'{k}={v:.4f}' for k, v in knn.items())
            logger.info(f'[eval] epoch {epoch} ({time.time()-t1:.1f}s) {knn_str}')

            if knn['_avg'] > best_knn:
                best_knn = knn['_avg']
                best_epoch = epoch
                p = save_ckpt(model, optimizer, epoch, args, tag='best')
                logger.info(f'[ckpt] new best avg_knn={best_knn:.4f} → {p}')

    save_ckpt(model, optimizer, args.num_epochs - 1, args, tag='last')
    logger.info(f'training done. best_avg_knn={best_knn:.4f} @ epoch {best_epoch}')


if __name__ == '__main__':
    main()
