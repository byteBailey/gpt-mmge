"""
统计 MAGB 5 个数据集的边 class consistency:
  对每条无向边 (u, v), 检查 y[u] == y[v] 是否成立, 计算占比.
  - 高 (>80%): 邻居信号干净, 当 anchor 找邻居"互相印证"是好策略 (Mario / R-MAGE 的边可靠性 motivation 弱化)
  - 中 (60~80%): 边带噪声, edge gating 有意义, 但不致命
  - 低 (<60%): 边噪声大, edge reliability ρ_uv 设计有强 motivation

也按 hop (1-hop / 2-hop) 分别统计, 看噪声是否随 hop 加剧.
"""

import json
from collections import Counter
from pathlib import Path

import torch
from torch_geometric.utils import to_undirected


GRAPH_PATHS = {
    'movies':  '/home/liyijun/gpt-mmge/MAGB/Movies/FineFeatures_convert/Movies_movie_ori_text_aug_imgText_ori_img_graph_data.pt',
    'toys':    '/home/liyijun/gpt-mmge/MAGB/Toys/FineFeatures_convert/Toys_toy_ori_text_aug_imgText_ori_img_graph_data.pt',
    'grocery': '/home/liyijun/gpt-mmge/MAGB/Grocery/FineFeatures_convert/Grocery_grocery_ori_text_aug_imgText_ori_img_graph_data.pt',
    'arts':    '/home/liyijun/gpt-mmge/MAGB/Arts/FineFeatures_convert/Arts_arts_ori_text_aug_imgText_ori_img_graph_data.pt',
    'cd':      '/home/liyijun/gpt-mmge/MAGB/CD/FineFeatures_convert/CD_cd_ori_text_aug_imgText_ori_img_graph_data.pt',
}


def two_hop_pairs(edge_index, num_nodes, max_pairs_per_node=20):
    """对每个节点 v 收集 2-hop 邻居对 (v, w), 截断防爆炸."""
    # 构邻接表
    adj = [[] for _ in range(num_nodes)]
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    for s, d in zip(src, dst):
        adj[s].append(d)
        adj[d].append(s)
    # 去重
    adj = [list(set(a)) for a in adj]

    pairs_src, pairs_dst = [], []
    import random
    random.seed(0)
    for v in range(num_nodes):
        # v 的 1-hop 邻居 -> 它们的 1-hop 邻居 (排除 v 自身和 v 的 1-hop)
        one_hop = set(adj[v])
        two_hop_set = set()
        for u in one_hop:
            for w in adj[u]:
                if w != v and w not in one_hop:
                    two_hop_set.add(w)
        if not two_hop_set:
            continue
        sampled = (random.sample(list(two_hop_set), max_pairs_per_node)
                   if len(two_hop_set) > max_pairs_per_node else list(two_hop_set))
        for w in sampled:
            pairs_src.append(v)
            pairs_dst.append(w)
    return torch.tensor(pairs_src, dtype=torch.long), torch.tensor(pairs_dst, dtype=torch.long)


def analyze(name, path):
    print(f'\n=== {name} ===')
    g = torch.load(path, weights_only=False)
    N = g.x.size(0) if hasattr(g, 'x') else len(g.y)
    y = g.y.long()
    n_classes = int(y.max().item()) + 1
    print(f'  nodes={N}, edges(directed)={g.edge_index.size(1)}, classes={n_classes}')

    # ── 1-hop ──
    # 转无向 (去重)
    eu = to_undirected(g.edge_index, num_nodes=N)
    src, dst = eu[0], eu[1]
    mask_no_self = src != dst
    src, dst = src[mask_no_self], dst[mask_no_self]
    # 因为 to_undirected 会同时保留 (u,v) 和 (v,u), 去 src<dst 的方向
    keep = src < dst
    src, dst = src[keep], dst[keep]
    n_edges = src.numel()

    same = (y[src] == y[dst])
    consistency_1hop = same.float().mean().item()
    print(f'  1-hop edges (undirected, no self-loop): {n_edges:,}')
    print(f'  1-hop class consistency (P[y_u==y_v]): {consistency_1hop*100:.2f}%')
    # 与随机基线对比
    label_freqs = torch.bincount(y, minlength=n_classes).float() / y.numel()
    random_baseline = (label_freqs ** 2).sum().item()
    print(f'  random baseline (Σ p_c^2):              {random_baseline*100:.2f}%')
    lift = consistency_1hop / random_baseline if random_baseline > 0 else float('inf')
    print(f'  homophily lift over random:             {lift:.2f}×')

    # ── 2-hop ──
    p2_src, p2_dst = two_hop_pairs(g.edge_index, N, max_pairs_per_node=20)
    same_2 = (y[p2_src] == y[p2_dst])
    consistency_2hop = same_2.float().mean().item()
    print(f'  2-hop pairs sampled (cap 20/node):       {p2_src.numel():,}')
    print(f'  2-hop class consistency:                 {consistency_2hop*100:.2f}%')

    # ── 邻居 label 多数派与 anchor 一致率 ──
    # 给每个节点, 统计其 1-hop 邻居 label 的众数 == y[v] 的比例
    # (即 GNN message passing 的"理论上限"信号)
    adj = [[] for _ in range(N)]
    for s, d in zip(g.edge_index[0].tolist(), g.edge_index[1].tolist()):
        if s != d:
            adj[s].append(d)
            adj[d].append(s)
    correct_majority = 0
    nodes_with_nbr = 0
    for v in range(N):
        if not adj[v]:
            continue
        nodes_with_nbr += 1
        cnt = Counter(y[u].item() for u in adj[v])
        majority = cnt.most_common(1)[0][0]
        if majority == y[v].item():
            correct_majority += 1
    majority_acc = correct_majority / max(1, nodes_with_nbr)
    print(f'  nodes with ≥1 neighbor: {nodes_with_nbr:,}')
    print(f'  anchor label == 1-hop neighbor majority: {majority_acc*100:.2f}%')

    return {
        'name': name,
        'nodes': N,
        'edges_undirected': n_edges,
        'classes': n_classes,
        '1hop_consistency': consistency_1hop,
        '2hop_consistency': consistency_2hop,
        'random_baseline': random_baseline,
        'homophily_lift': lift,
        'majority_acc': majority_acc,
    }


def main():
    results = []
    for name, path in GRAPH_PATHS.items():
        try:
            r = analyze(name, path)
            results.append(r)
        except Exception as e:
            print(f'[error] {name}: {e}')

    print('\n\n========== SUMMARY ==========')
    print(f'{"dataset":<10} | {"nodes":>7} | {"edges":>9} | {"cls":>3} | '
          f'{"1hop%":>6} | {"2hop%":>6} | {"rand%":>6} | {"lift":>5} | {"maj%":>6}')
    print('-' * 90)
    for r in results:
        print(f'{r["name"]:<10} | {r["nodes"]:>7,} | {r["edges_undirected"]:>9,} | {r["classes"]:>3} | '
              f'{r["1hop_consistency"]*100:>5.1f}% | {r["2hop_consistency"]*100:>5.1f}% | '
              f'{r["random_baseline"]*100:>5.1f}% | {r["homophily_lift"]:>4.1f}× | '
              f'{r["majority_acc"]*100:>5.1f}%')

    # 写 json
    out_path = Path('/home/liyijun/gpt-mmge/gpt-mmge-v8/edge_consistency.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nresults saved to {out_path}')


if __name__ == '__main__':
    main()
