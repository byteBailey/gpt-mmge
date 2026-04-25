import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dgl.data.utils import load_graphs
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data


RAW_ROOT = Path("/home/liyijun/gpt-mmge/MAGB/Toys")
OUT_DIR = Path("/home/liyijun/gpt-mmge/MAGB/Toys/FineFeatures_convert")

csv_path = RAW_ROOT / "Toys.csv"
text_feat_path = RAW_ROOT / "TextFeature" / "Toys_roberta_base_512_mean.npy"
img_feat_path = RAW_ROOT / "ImageFeature" / "Toys_openai_clip-vit-large-patch14.npy"
graph_pt_path = RAW_ROOT / "ToysGraph.pt"
fine_feature_dir = RAW_ROOT / "FineFeatures"  # 细粒度特征目录 (由 extract_fine_features.py 生成)


def build_raw_text(row):
    text = row.get("text", None)
    if isinstance(text, str) and text.strip():
        return text

    title = row.get("title", "")
    desc = row.get("description", "")

    title = "" if pd.isna(title) else str(title)
    desc = "" if pd.isna(desc) else str(desc)

    return f"Title: {title}; Description: {desc}"


def to_index_list(obj, expected_num_nodes=None):
    if obj is None:
        return None

    if isinstance(obj, (list, tuple)):
        return [int(x) for x in obj]

    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)

    if torch.is_tensor(obj):
        obj = obj.cpu()

        if obj.dtype == torch.bool:
            return torch.where(obj)[0].tolist()

        obj = obj.view(-1)
        if expected_num_nodes is not None and obj.numel() == expected_num_nodes:
            uniq = set(obj.unique().tolist())
            if uniq.issubset({0, 1}):
                return torch.where(obj.bool())[0].tolist()

        return obj.long().tolist()

    raise TypeError(f"Unsupported split object type: {type(obj)}")


def get_split_indices(g_raw, label_dict):
    num_nodes = g_raw.num_nodes()

    print("==== DGL graph info ====")
    print("num_nodes:", g_raw.num_nodes())
    print("num_edges:", g_raw.num_edges())
    print("ndata keys:", list(g_raw.ndata.keys()))
    print("edata keys:", list(g_raw.edata.keys()))
    print("label_dict keys:", list(label_dict.keys()))

    ndata_keys = set(g_raw.ndata.keys())
    if {"train_mask", "val_mask", "test_mask"}.issubset(ndata_keys):
        train_ids = to_index_list(g_raw.ndata["train_mask"], expected_num_nodes=num_nodes)
        val_ids = to_index_list(g_raw.ndata["val_mask"], expected_num_nodes=num_nodes)
        test_ids = to_index_list(g_raw.ndata["test_mask"], expected_num_nodes=num_nodes)
        print("split source: g_raw.ndata[train_mask/val_mask/test_mask]")
        return train_ids, val_ids, test_ids

    if {"train_idx", "val_idx", "test_idx"}.issubset(ndata_keys):
        train_ids = to_index_list(g_raw.ndata["train_idx"], expected_num_nodes=num_nodes)
        val_ids = to_index_list(g_raw.ndata["val_idx"], expected_num_nodes=num_nodes)
        test_ids = to_index_list(g_raw.ndata["test_idx"], expected_num_nodes=num_nodes)
        print("split source: g_raw.ndata[train_idx/val_idx/test_idx]")
        return train_ids, val_ids, test_ids

    label_keys = set(label_dict.keys())
    if {"train_mask", "val_mask", "test_mask"}.issubset(label_keys):
        train_ids = to_index_list(label_dict["train_mask"], expected_num_nodes=num_nodes)
        val_ids = to_index_list(label_dict["val_mask"], expected_num_nodes=num_nodes)
        test_ids = to_index_list(label_dict["test_mask"], expected_num_nodes=num_nodes)
        print("split source: label_dict[train_mask/val_mask/test_mask]")
        return train_ids, val_ids, test_ids

    if {"train_idx", "val_idx", "test_idx"}.issubset(label_keys):
        train_ids = to_index_list(label_dict["train_idx"], expected_num_nodes=num_nodes)
        val_ids = to_index_list(label_dict["val_idx"], expected_num_nodes=num_nodes)
        test_ids = to_index_list(label_dict["test_idx"], expected_num_nodes=num_nodes)
        print("split source: label_dict[train_idx/val_idx/test_idx]")
        return train_ids, val_ids, test_ids

    return None, None, None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 检查细粒度特征是否存在 ──
    use_fine = fine_feature_dir.exists() and (fine_feature_dir / "img_features.pt").exists()

    # 基础必须文件: CSV + Graph
    required_files = [csv_path, graph_pt_path]
    # 细粒度特征不存在时, 才要求 MAGB 单向量 .npy 文件
    if not use_fine:
        required_files += [text_feat_path, img_feat_path]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing Toys raw files:\n" + "\n".join(missing_files)
        )

    df = pd.read_csv(csv_path)
    y = torch.tensor(df["label"].to_numpy(), dtype=torch.long)
    raw_texts = [build_raw_text(row) for _, row in df.iterrows()]

    label_df = (
        df[["label", "second_category"]]
        .drop_duplicates()
        .sort_values("label")
        .reset_index(drop=True)
    )
    label_texts = label_df["second_category"].tolist()

    graphs, label_dict = load_graphs(str(graph_pt_path))
    if len(graphs) == 0:
        raise ValueError(f"No graph found in {graph_pt_path}")

    g_raw = graphs[0]
    if g_raw.num_nodes() != len(df):
        raise ValueError(
            f"Graph node count does not match CSV rows: num_nodes={g_raw.num_nodes()}, len(df)={len(df)}"
        )

    src, dst = g_raw.edges()
    edge_index = torch.stack([src, dst], dim=0).long()

    # ── 多模态特征: 优先使用细粒度特征, 否则回退到 MAGB 单向量 ──
    if use_fine:
        print(f"Loading fine-grained features from {fine_feature_dir}")
        img_features = torch.load(fine_feature_dir / "img_features.pt", weights_only=True).float()
        txt_features = torch.load(fine_feature_dir / "txt_features.pt", weights_only=True).float()
        img_mask = torch.load(fine_feature_dir / "img_mask.pt", weights_only=True)
        txt_mask = torch.load(fine_feature_dir / "txt_mask.pt", weights_only=True)
        print(f"  img_features: {img_features.shape}, txt_features: {txt_features.shape}")
        # x: 从细粒度特征的 CLS/mean 拼接得到
        img_cls = img_features[:, 0, :]  # [N, img_dim]
        txt_mean = (txt_features * txt_mask.unsqueeze(-1)).sum(dim=1) / txt_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [N, txt_dim]
        x = torch.cat([txt_mean, img_cls], dim=1)
    else:
        print("Fine-grained features not found, using MAGB single-vector features")
        text_feat = np.load(text_feat_path)
        img_feat = np.load(img_feat_path)
        assert len(df) == text_feat.shape[0] == img_feat.shape[0], (
            len(df), text_feat.shape, img_feat.shape
        )
        x = torch.tensor(np.concatenate([text_feat, img_feat], axis=1), dtype=torch.float32)
        img_features = torch.tensor(img_feat, dtype=torch.float32).unsqueeze(1)  # [N, 1, 768]
        txt_features = torch.tensor(text_feat, dtype=torch.float32).unsqueeze(1)  # [N, 1, 768]
        img_mask = torch.ones(len(df), 1, dtype=torch.bool)
        txt_mask = torch.ones(len(df), 1, dtype=torch.bool)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.raw_texts = raw_texts
    data.label_texts = label_texts
    data.img_features = img_features
    data.txt_features = txt_features
    data.img_mask = img_mask
    data.txt_mask = txt_mask
    data.edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long)

    out_graph_path = OUT_DIR / "Toys_toy_ori_text_aug_imgText_ori_img_graph_data.pt"
    torch.save(data, out_graph_path)

    train_ids, val_ids, test_ids = get_split_indices(g_raw, label_dict)
    if train_ids is None:
        print("No predefined split found in graph. Generate stratified 60/20/20 split from CSV labels.")

        all_idx = np.arange(len(df))
        all_labels = df["label"].to_numpy()

        train_idx, temp_idx, train_y, temp_y = train_test_split(
            all_idx,
            all_labels,
            test_size=0.4,
            random_state=42,
            stratify=all_labels,
        )
        val_idx, test_idx, _, _ = train_test_split(
            temp_idx,
            temp_y,
            test_size=0.5,
            random_state=42,
            stratify=temp_y,
        )

        train_ids = train_idx.tolist()
        val_ids = val_idx.tolist()
        test_ids = test_idx.tolist()

        print(f"Generated split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    split_path = OUT_DIR / "Toys_split.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(
            {"train": train_ids, "val": val_ids, "test": test_ids},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n==== Saved ====")
    print("graph data:", out_graph_path)
    print("split json:", split_path)
    print("x shape:", data.x.shape)
    print("edge_index shape:", data.edge_index.shape)
    print("y shape:", data.y.shape)
    print("num labels:", len(data.label_texts))
    print("train/val/test:", len(train_ids), len(val_ids), len(test_ids))
    print("img_features shape:", data.img_features.shape)
    print("txt_features shape:", data.txt_features.shape)
    print("edge_type shape:", data.edge_type.shape)


if __name__ == "__main__":
    main()
