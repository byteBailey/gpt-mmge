"""
细粒度特征提取: 从原始图片和文本描述中提取 patch-level / token-level 特征。

图片: CLIP ViT-L/14 → [N, 257, 1024]  (CLS + 256 spatial patches)
文本: RoBERTa-base  → [N, 256, 768] + attention_mask [N, 256]  (覆盖 94.8% 的文本)

用法:
    python -m extract_fine_features \
        --dataset_dir /home/liyijun/gpt-mmge/MAGB/CD \
        --image_dir /home/liyijun/gpt-mmge/MAGB/CD/CDSImages \
        --clip_model /nfs/llm-models/clip-vit-large-patch14 \
        --text_model /nfs/llm-models/roberta-base \
        --max_text_len 256 \
        --batch_size 64
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


def extract_image_features(
    image_dir: Path,
    node_ids: list,
    clip_model_name: str,
    batch_size: int,
    device: str,
):
    """
    用 CLIP Vision Encoder 提取 patch-level 特征。
    返回 img_features [N, 257, 1024] 和 img_mask [N, 257]。
    缺失图片的节点用零填充, mask 置 0。
    """
    from transformers import CLIPVisionModel, CLIPProcessor

    print(f'Loading CLIP vision model: {clip_model_name}')
    model = CLIPVisionModel.from_pretrained(clip_model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(clip_model_name)

    # 获取模型输出维度: 先跑一张虚拟图片探测
    dummy = Image.new('RGB', (224, 224))
    dummy_inputs = processor(images=dummy, return_tensors='pt').to(device)
    with torch.no_grad():
        dummy_out = model(**dummy_inputs)
    seq_len = dummy_out.last_hidden_state.shape[1]  # 257 for ViT-L/14
    hidden_dim = dummy_out.last_hidden_state.shape[2]  # 1024
    print(f'  CLIP output: seq_len={seq_len}, hidden_dim={hidden_dim}')

    N = len(node_ids)
    img_features = torch.zeros(N, seq_len, hidden_dim, dtype=torch.float16)
    img_mask = torch.zeros(N, seq_len, dtype=torch.bool)

    # 按 batch 提取
    for start in tqdm(range(0, N, batch_size), desc='Extracting image features'):
        end = min(start + batch_size, N)
        batch_ids = node_ids[start:end]

        images = []
        valid_indices = []  # 在当前 batch 内的索引
        for j, nid in enumerate(batch_ids):
            img_path = image_dir / f'{nid}.jpg'
            if img_path.exists():
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                    valid_indices.append(j)
                except Exception as e:
                    print(f'  Warning: failed to load {img_path}: {e}')

        if not images:
            continue

        inputs = processor(images=images, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        feats = outputs.last_hidden_state.cpu().half()  # [B_valid, seq_len, hidden_dim]

        for k, j in enumerate(valid_indices):
            global_idx = start + j
            img_features[global_idx] = feats[k]
            img_mask[global_idx] = True

    num_valid = img_mask[:, 0].sum().item()
    print(f'  Image features extracted: {num_valid}/{N} nodes have images')
    return img_features, img_mask


def extract_text_features(
    texts: list,
    text_model_name: str,
    max_text_len: int,
    batch_size: int,
    device: str,
):
    """
    用 RoBERTa 提取 token-level 特征。
    返回 txt_features [N, max_text_len, 768] 和 txt_mask [N, max_text_len]。
    """
    from transformers import RobertaModel, RobertaTokenizer

    print(f'Loading text model: {text_model_name}')
    tokenizer = RobertaTokenizer.from_pretrained(text_model_name)
    model = RobertaModel.from_pretrained(text_model_name).to(device).eval()

    hidden_dim = model.config.hidden_size  # 768
    print(f'  Text output: max_len={max_text_len}, hidden_dim={hidden_dim}')

    N = len(texts)
    txt_features = torch.zeros(N, max_text_len, hidden_dim, dtype=torch.float16)
    txt_mask = torch.zeros(N, max_text_len, dtype=torch.bool)

    for start in tqdm(range(0, N, batch_size), desc='Extracting text features'):
        end = min(start + batch_size, N)
        batch_texts = texts[start:end]

        inputs = tokenizer(
            batch_texts,
            max_length=max_text_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        feats = outputs.last_hidden_state.cpu().half()  # [B, max_len, hidden_dim]
        masks = inputs['attention_mask'].cpu().bool()    # [B, max_len]

        txt_features[start:end] = feats
        txt_mask[start:end] = masks

    print(f'  Text features extracted for {N} nodes')
    return txt_features, txt_mask


def main():
    parser = argparse.ArgumentParser(description='Extract fine-grained features')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Path to dataset dir (e.g. MAGB/Grocery)')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Path to image dir (e.g. MAGB/Grocery/GrocerySImages)')
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-large-patch14')
    parser.add_argument('--text_model', type=str, default='roberta-base')
    parser.add_argument('--max_text_len', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    image_dir = Path(args.image_dir)
    csv_path = dataset_dir / 'CD.csv'

    print(f'Reading CSV: {csv_path}')
    df = pd.read_csv(csv_path)
    N = len(df)
    node_ids = df['id'].tolist()
    print(f'  Total nodes: {N}')

    # 构建文本: 优先使用 text 列, 否则拼接 title + description
    texts = []
    for _, row in df.iterrows():
        text = row.get('text', None)
        if isinstance(text, str) and text.strip():
            texts.append(text)
        else:
            title = '' if pd.isna(row.get('title', '')) else str(row['title'])
            desc = '' if pd.isna(row.get('description', '')) else str(row['description'])
            texts.append(f'Title: {title}; Description: {desc}')

    # 提取图片特征
    img_features, img_mask = extract_image_features(
        image_dir, node_ids, args.clip_model, args.batch_size, args.device,
    )

    # 提取文本特征
    txt_features, txt_mask = extract_text_features(
        texts, args.text_model, args.max_text_len, args.batch_size, args.device,
    )

    # 保存
    out_dir = dataset_dir / 'FineFeatures'
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(img_features, out_dir / 'img_features.pt')
    torch.save(img_mask, out_dir / 'img_mask.pt')
    torch.save(txt_features, out_dir / 'txt_features.pt')
    torch.save(txt_mask, out_dir / 'txt_mask.pt')

    print(f'\nSaved to {out_dir}:')
    print(f'  img_features: {img_features.shape} ({img_features.dtype})')
    print(f'  img_mask:     {img_mask.shape}')
    print(f'  txt_features: {txt_features.shape} ({txt_features.dtype})')
    print(f'  txt_mask:     {txt_mask.shape}')


if __name__ == '__main__':
    main()
