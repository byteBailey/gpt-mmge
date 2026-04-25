"""
Stage 1 多数据集联合预训练 Dataset。

把 5 个 MAGB 数据集 (Movies/Toys/Grocery/Arts/CD) 包装成一个统一 Dataset。
每条样本: 全局 idx → (dataset_id, anchor_node_id, positive_neighbor_id)。
预先为每个数据集构建 SubgraphBuilder, 运行时按 anchor/positive 分别 build k-hop 子图。

Collate: 把 2*B 个子图 (B 个 anchor + B 个 positive) 用 collate_subgraphs() 合并,
        encoder 一次性处理, 取每个子图的 center 节点形成 [2B, K, D]。
"""

import json
import random
import torch
from torch.utils.data import Dataset

from src.dataset.subgraph_builder import SubgraphBuilder
from src.model.multimodal_graph_encoder import collate_subgraphs


GRAPH_PATHS = {
    'movies':  '/home/liyijun/gpt-mmge/MAGB/Movies/FineFeatures_convert/Movies_movie_ori_text_aug_imgText_ori_img_graph_data.pt',
    'toys':    '/home/liyijun/gpt-mmge/MAGB/Toys/FineFeatures_convert/Toys_toy_ori_text_aug_imgText_ori_img_graph_data.pt',
    # 小版 Grocery (17k 节点); 大版在 MAGB/Grocery_84k (84k 节点, 当前不用)
    'grocery': '/home/liyijun/gpt-mmge/MAGB/Grocery/FineFeatures_convert/Grocery_grocery_ori_text_aug_imgText_ori_img_graph_data.pt',
    'arts':    '/home/liyijun/gpt-mmge/MAGB/Arts/FineFeatures_convert/Arts_arts_ori_text_aug_imgText_ori_img_graph_data.pt',
    'cd':      '/home/liyijun/gpt-mmge/MAGB/CD/FineFeatures_convert/CD_cd_ori_text_aug_imgText_ori_img_graph_data.pt',
}

SPLIT_PATHS = {
    'movies':  '/home/liyijun/gpt-mmge/MAGB/Movies/FineFeatures_convert/Movies_split.json',
    'toys':    '/home/liyijun/gpt-mmge/MAGB/Toys/FineFeatures_convert/Toys_split.json',
    'grocery': '/home/liyijun/gpt-mmge/MAGB/Grocery/FineFeatures_convert/Grocery_split.json',
    'arts':    '/home/liyijun/gpt-mmge/MAGB/Arts/FineFeatures_convert/Arts_split.json',
    'cd':      '/home/liyijun/gpt-mmge/MAGB/CD/FineFeatures_convert/CD_split.json',
}


def load_graphs_and_splits(dataset_names):
    graphs, splits = [], []
    for name in dataset_names:
        print(f'[pretrain] loading {name} ...')
        graph = torch.load(GRAPH_PATHS[name])
        with open(SPLIT_PATHS[name], 'r', encoding='utf-8') as f:
            split = json.load(f)
        for k in ('train', 'val', 'test'):
            split[k] = [int(i) for i in split[k]]
        print(f'  {name}: nodes={graph.x.size(0)} edges={graph.edge_index.size(1)} '
              f'img={tuple(graph.img_features.shape)} txt={tuple(graph.txt_features.shape)}')
        graphs.append(graph)
        splits.append(split)
    return graphs, splits


class MMGPretrainDataset(Dataset):
    """跨多个 MAGB 图的联合预训练数据集。

    全局 idx 映射: idx ∈ [cumsum[i], cumsum[i+1]) → 数据集 i 的 train_ids 中的某个节点。
    只在每个数据集的 train split 上做预训练 (避免污染 val/test)。
    """

    def __init__(self, graphs, splits, num_hops=2, max_neighbors=20):
        super().__init__()
        self.graphs = graphs
        self.splits = splits
        self.train_ids_per_ds = [s['train'] for s in splits]
        self.builders = [SubgraphBuilder(g, max_neighbors=max_neighbors, num_hops=num_hops)
                         for g in graphs]
        sizes = [len(t) for t in self.train_ids_per_ds]
        self.cumsum = [0]
        for s in sizes:
            self.cumsum.append(self.cumsum[-1] + s)
        print(f'[pretrain] joint train set: {self.cumsum[-1]} anchors '
              f'(per-dataset sizes: {sizes})')

    def __len__(self):
        return self.cumsum[-1]

    def _locate(self, global_idx):
        for i in range(len(self.cumsum) - 1):
            if global_idx < self.cumsum[i + 1]:
                return i, global_idx - self.cumsum[i]
        raise IndexError(global_idx)

    def __getitem__(self, global_idx):
        ds_id, local_pos = self._locate(global_idx)
        anchor_id = self.train_ids_per_ds[ds_id][local_pos]

        builder = self.builders[ds_id]
        nbrs = builder.adj_nbr[anchor_id]
        pos_id = random.choice(nbrs) if len(nbrs) > 0 else anchor_id

        anchor_sg = builder.build_one(anchor_id)
        pos_sg = builder.build_one(pos_id)

        return {
            'dataset_id': ds_id,
            'anchor_id': anchor_id,
            'pos_id': pos_id,
            'anchor_sg': anchor_sg,
            'pos_sg': pos_sg,
        }


def pretrain_collate(batch):
    """把 B 个样本的 (anchor, positive) 子图合并成一个大图, 共 2B 个子图。

    返回:
        batched: 直接 ** 进 MultimodalGraphEncoder.forward
        center_indices: [2B] 每个子图 center 在合并图中的全局索引
                        前 B 个是 anchor center, 后 B 个是 positive center
        batch_size: B
    """
    subgraphs = [item['anchor_sg'] for item in batch] + \
                [item['pos_sg'] for item in batch]
    batched, _ = collate_subgraphs(subgraphs)

    sizes = [sg['img_features'].size(0) for sg in subgraphs]
    offsets = [0]
    for s in sizes:
        offsets.append(offsets[-1] + s)
    center_indices = torch.tensor(offsets[:-1], dtype=torch.long)

    return {
        'batched': batched,
        'center_indices': center_indices,
        'batch_size': len(batch),
        'dataset_ids': [item['dataset_id'] for item in batch],
        'anchor_ids': [item['anchor_id'] for item in batch],
        'pos_ids': [item['pos_id'] for item in batch],
    }


class MMGEvalDataset(Dataset):
    """单数据集的评测 Dataset, 给定节点 id 列表, 输出每个节点的 k-hop 子图。"""

    def __init__(self, graph, node_ids, builder):
        self.graph = graph
        self.node_ids = list(node_ids)
        self.builder = builder

    def __len__(self):
        return len(self.node_ids)

    def __getitem__(self, idx):
        nid = self.node_ids[idx]
        sg = self.builder.build_one(nid)
        return {'node_id': nid, 'sg': sg, 'label': int(self.graph.y[nid])}


def eval_collate(batch):
    subgraphs = [item['sg'] for item in batch]
    batched, _ = collate_subgraphs(subgraphs)
    sizes = [sg['img_features'].size(0) for sg in subgraphs]
    offsets = [0]
    for s in sizes:
        offsets.append(offsets[-1] + s)
    center_indices = torch.tensor(offsets[:-1], dtype=torch.long)
    return {
        'batched': batched,
        'center_indices': center_indices,
        'node_ids': [item['node_id'] for item in batch],
        'labels': torch.tensor([item['label'] for item in batch], dtype=torch.long),
    }
