"""
Online k-hop ego-subgraph construction with induced edges.

Given a full graph (PyG Data with img_features, txt_features, edge_type, etc.)
and a list of center node IDs, build per-node k-hop subgraphs in the format
expected by MultimodalGraphEncoder, then collate via collate_subgraphs().

设计要点:
  - 初始化时预计算邻接表，避免运行时反复扫描 edge_index
  - 中心节点固定在局部索引 0
  - k-hop 采样: 与 GraphPrompter TAGCollator 相同的 per-frontier 总量截断
    每跳收集整个 frontier 的所有邻居, 总量超过 max_neighbors 时随机截断
  - 采样完成后建 induced subgraph: 保留子图节点间的所有原始边
  - 每次调用结果不同，起到数据增强 / 正则化效果
"""

import random
import torch
from typing import List


def _collate_subgraphs(subgraphs):
    # 延迟 import, 避免 stage 1 预训练时触发 src.model.__init__ 的循环导入
    from src.model.multimodal_graph_encoder import collate_subgraphs as _fn
    return _fn(subgraphs)


# 兼容旧代码中的本模块级符号
collate_subgraphs = _collate_subgraphs


class SubgraphBuilder:
    """Precomputes adjacency from full graph for fast k-hop lookup."""

    def __init__(self, graph, max_neighbors: int = 50, num_hops: int = 3):
        self.graph = graph
        self.max_neighbors = max_neighbors
        self.num_hops = num_hops

        # ── 预计算无向邻接表: node_id → [neighbor_id, ...] 和对应 edge_type ──
        src_list = graph.edge_index[0].tolist()
        dst_list = graph.edge_index[1].tolist()
        has_edge_type = hasattr(graph, 'edge_type') and graph.edge_type is not None
        etypes = graph.edge_type.tolist() if has_edge_type else [0] * len(src_list)

        num_nodes = graph.x.size(0)
        # adj_nbr[i] = [nbr1, nbr2, ...], adj_etype[i] = [e1, e2, ...]
        adj_nbr: List[List[int]] = [[] for _ in range(num_nodes)]
        adj_etype: List[List[int]] = [[] for _ in range(num_nodes)]
        for s, d, e in zip(src_list, dst_list, etypes):
            adj_nbr[s].append(d)
            adj_etype[s].append(e)
            adj_nbr[d].append(s)
            adj_etype[d].append(e)

        # 去重 (去除自环和重复邻居)
        for i in range(num_nodes):
            seen = set()
            new_nbr, new_etype = [], []
            for nbr, etype in zip(adj_nbr[i], adj_etype[i]):
                if nbr != i and nbr not in seen:
                    seen.add(nbr)
                    new_nbr.append(nbr)
                    new_etype.append(etype)
            adj_nbr[i] = new_nbr
            adj_etype[i] = new_etype

        self.adj_nbr = adj_nbr
        self.adj_etype = adj_etype
        # 快速查边类型: (src, dst) → etype
        self._edge_type_map = {}
        for i in range(num_nodes):
            for nbr, etype in zip(adj_nbr[i], adj_etype[i]):
                self._edge_type_map[(i, nbr)] = etype

    def build_one(self, center_id: int) -> dict:
        """
        Build k-hop ego subgraph with per-frontier random sampling + induced edges.
        采样方式与 GraphPrompter TAGCollator 的 batch_subgraph 完全一致:
          每跳收集整个 frontier 的所有邻居 → 总量超过 max_neighbors 时随机截断。
        Center node is always at local index 0.
        每次调用随机采样结果不同，起到数据增强效果。
        """
        max_n = self.max_neighbors
        graph = self.graph

        # ── Step 1: k-hop per-frontier random sampling (与 TAGCollator 一致) ──
        subsets = [{center_id}]
        frontier = {center_id}

        for hop in range(self.num_hops):
            # 收集整个 frontier 的所有邻居 (去掉自身)
            all_neighbors = []
            for node in frontier:
                all_neighbors.extend(self.adj_nbr[node])
            # 总量截断 (对整个 frontier 的邻居集合做一次截断，不是 per-node)
            if len(all_neighbors) > max_n:
                all_neighbors = random.sample(all_neighbors, max_n)
            # 去重并加入已访问集合
            new_nodes = set(all_neighbors)
            subsets.append(new_nodes)
            frontier = new_nodes

        # 合并所有跳的节点并去重
        visited = set()
        for s in subsets:
            visited.update(s)

        # ── Step 2: 局部节点排序, center 固定在 idx=0 ──
        local_nodes = [center_id] + [n for n in visited if n != center_id]
        num_local = len(local_nodes)
        local_set = set(local_nodes)
        global2local = {gid: lid for lid, gid in enumerate(local_nodes)}

        local_nodes_t = torch.tensor(local_nodes, dtype=torch.long)

        # 特征 (形状取决于数据: 单向量 [M, 1, D] 或细粒度 [M, P, D])
        img_features = graph.img_features[local_nodes_t]           # [M, P, img_dim]
        txt_features = graph.txt_features[local_nodes_t]           # [M, T, txt_dim]
        img_mask = graph.img_mask[local_nodes_t].float()           # [M, P]
        txt_mask = graph.txt_mask[local_nodes_t].float()           # [M, T]

        # ── Step 3: 构建 induced subgraph 的 neighbor_index ──
        # 对每个子图内节点, 保留它在原图中所有也在子图内的邻居 (induced edges)
        # 超过 max_n 时随机截断
        neighbor_index = torch.zeros(num_local, max_n, dtype=torch.long)
        rel_ids = torch.zeros(num_local, max_n, dtype=torch.long)
        neighbor_mask = torch.zeros(num_local, max_n, dtype=torch.float)

        for local_idx, global_id in enumerate(local_nodes):
            # 收集该节点在子图内的所有邻居 (induced)
            induced_nbrs = []
            for nbr, etype in zip(self.adj_nbr[global_id], self.adj_etype[global_id]):
                if nbr in local_set and nbr != global_id:
                    induced_nbrs.append((nbr, etype))
            # 超过上限时随机截断
            if len(induced_nbrs) > max_n:
                induced_nbrs = random.sample(induced_nbrs, max_n)
            for pos, (nbr_gid, etype) in enumerate(induced_nbrs):
                neighbor_index[local_idx, pos] = global2local[nbr_gid]
                rel_ids[local_idx, pos] = etype
                neighbor_mask[local_idx, pos] = 1.0

        return {
            "img_features": img_features,
            "txt_features": txt_features,
            "neighbor_index": neighbor_index,
            "rel_ids": rel_ids,
            "neighbor_mask": neighbor_mask,
            "img_mask": img_mask,
            "txt_mask": txt_mask,
        }

    # ────────────────────────────────────────────────────────────
    # LLaGA-style: 固定长度、按跳分层的节点序列
    # ────────────────────────────────────────────────────────────

    def _sample_fixed(self, node_id: int, sample_size: int):
        """对 node_id 采样恰好 sample_size 个邻居, 不足补 -1 (PAD)。"""
        nbrs = self.adj_nbr[node_id]
        if len(nbrs) >= sample_size:
            return random.sample(nbrs, sample_size)
        elif len(nbrs) > 0:
            return list(nbrs) + [-1] * (sample_size - len(nbrs))
        else:
            return [-1] * sample_size

    def build_one_seq(self, center_id: int, sample_size: int = 10) -> dict:
        """
        Build k-hop ego subgraph + fixed-length hop sequence.

        树形采样: 每个节点固定采 sample_size 个邻居, 不足补 PAD。
        序列长度 = sum(sample_size^i for i in 0..num_hops)
          hop=1, S=10 → 11
          hop=2, S=10 → 111
          hop=2, S=5  → 31

        返回 dict 包含:
          - 标准子图字段 (img_features, txt_features, neighbor_index 等)
          - hop_seq: [seq_len] 每个位置对应的局部节点索引, -1 表示 PAD
        """
        graph = self.graph
        max_n = self.max_neighbors

        # ── Step 1: 树形固定采样 ──
        PAD = -1
        seq_global = [center_id]          # hop 0: center

        prev_hop = [center_id]
        for hop in range(self.num_hops):
            current_hop = []
            for parent in prev_hop:
                if parent == PAD:
                    current_hop.extend([PAD] * sample_size)
                else:
                    current_hop.extend(self._sample_fixed(parent, sample_size))
            seq_global.extend(current_hop)
            prev_hop = current_hop

        # ── Step 2: 收集去重的真实节点 (center 在 idx=0) ──
        unique_real = []
        seen = set()
        for gid in seq_global:
            if gid != PAD and gid not in seen:
                unique_real.append(gid)
                seen.add(gid)

        local_nodes = unique_real
        num_local = len(local_nodes)
        local_set = set(local_nodes)
        global2local = {gid: lid for lid, gid in enumerate(local_nodes)}

        # ── Step 3: hop_seq 映射 ──
        hop_seq = torch.tensor(
            [global2local[gid] if gid != PAD else -1 for gid in seq_global],
            dtype=torch.long
        )

        # ── Step 4: 子图特征 ──
        local_nodes_t = torch.tensor(local_nodes, dtype=torch.long)
        img_features = graph.img_features[local_nodes_t]
        txt_features = graph.txt_features[local_nodes_t]
        img_mask = graph.img_mask[local_nodes_t].float()
        txt_mask = graph.txt_mask[local_nodes_t].float()

        # ── Step 5: Induced subgraph 的邻接关系 ──
        neighbor_index = torch.zeros(num_local, max_n, dtype=torch.long)
        rel_ids = torch.zeros(num_local, max_n, dtype=torch.long)
        neighbor_mask = torch.zeros(num_local, max_n, dtype=torch.float)

        for local_idx, global_id in enumerate(local_nodes):
            induced_nbrs = []
            for nbr, etype in zip(self.adj_nbr[global_id], self.adj_etype[global_id]):
                if nbr in local_set and nbr != global_id:
                    induced_nbrs.append((nbr, etype))
            if len(induced_nbrs) > max_n:
                induced_nbrs = random.sample(induced_nbrs, max_n)
            for pos, (nbr_gid, etype) in enumerate(induced_nbrs):
                neighbor_index[local_idx, pos] = global2local[nbr_gid]
                rel_ids[local_idx, pos] = etype
                neighbor_mask[local_idx, pos] = 1.0

        return {
            "img_features": img_features,
            "txt_features": txt_features,
            "neighbor_index": neighbor_index,
            "rel_ids": rel_ids,
            "neighbor_mask": neighbor_mask,
            "img_mask": img_mask,
            "txt_mask": txt_mask,
            "hop_seq": hop_seq,       # [seq_len] 新增: 序列→局部索引映射
        }

    def build_batch_seq(self, center_ids, sample_size: int = 10):
        """
        Build and collate k-hop ego subgraphs with fixed-length hop sequences.

        Returns:
            batched: dict for MultimodalGraphEncoder.forward(**batched)
            batch_vec: [M_total]
            center_indices: [B] center node in merged graph
            hop_seqs: [B, seq_len] global indices in merged graph, -1 for PAD
        """
        subgraphs = [self.build_one_seq(int(cid), sample_size) for cid in center_ids]

        # 取出 hop_seq 并记录各子图节点数
        hop_seqs = []
        sizes = []
        for sg in subgraphs:
            hop_seqs.append(sg.pop("hop_seq"))
            sizes.append(sg["img_features"].size(0))

        batched, batch_vec = collate_subgraphs(subgraphs)

        # hop_seq 中的局部索引加上全局偏移
        offset = 0
        adjusted_seqs = []
        center_indices = []
        for seq, sz in zip(hop_seqs, sizes):
            center_indices.append(offset)
            adj = seq.clone()
            adj[adj >= 0] += offset
            adjusted_seqs.append(adj)
            offset += sz

        hop_seqs = torch.stack(adjusted_seqs)       # [B, seq_len]
        center_indices = torch.tensor(center_indices, dtype=torch.long)

        return batched, batch_vec, center_indices, hop_seqs

    # ────────────────────────────────────────────────────────────
    # 原始方式: 变长子图, 仅返回中心节点
    # ────────────────────────────────────────────────────────────

    def build_batch(self, center_ids):
        """
        Build and collate k-hop ego subgraphs for a batch of center nodes.

        Returns:
            batched: dict ready for MultimodalGraphEncoder.forward(**batched)
            batch_vec: [M_total] which subgraph each entity belongs to
            center_indices: [B] index of each center node in the merged graph
        """
        subgraphs = [self.build_one(int(cid)) for cid in center_ids]
        batched, batch_vec = collate_subgraphs(subgraphs)

        # 每个子图的中心节点在局部索引 0，累加偏移量得到全局索引
        center_indices = []
        offset = 0
        for sg in subgraphs:
            center_indices.append(offset)
            offset += sg["img_features"].size(0)
        center_indices = torch.tensor(center_indices, dtype=torch.long)

        return batched, batch_vec, center_indices
