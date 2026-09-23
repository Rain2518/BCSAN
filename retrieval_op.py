"""
OPMMD / RSICD Image-Text Retrieval Baseline using OpenCLIP
===========================================================
标准图文检索流程（论文级 baseline）：
1. 加载 OpenCLIP 模型
2. 加载 OPMMD/RSICD 数据集
3. 提取唯一图片特征 (U x D)   ——  每张图片编码一次，不复制
4. 提取所有 caption 特征 (M x D)
5. 计算相似度矩阵: I2T (U x M), T2I (M x U)
6. 计算 Recall@K (K=1,5,10)

用法 (OPMMD):
    python retrieval_op.py \
        --model ViT-B-32 \
        --pretrained /root/autodl-tmp/ViT-B-32.pt \
        --image_root /root/autodl-tmp/OPMMD/image \
        --image_subdir test \
        --caption_file /root/autodl-tmp/dataset/test.json \
        --batch_size 64 \
        --device cuda

用法 (RSICD):
    python retrieval_op.py \
        --model ViT-B-16 \
        --pretrained /path/to/model.bin \
        --data_root /path/to/RSICD/images \
        --image_dir . \
        --caption_file /path/to/dataset_rsicd.json \
        --batch_size 64 \
        --device cuda

数据集格式支持:
    格式1 (简单字典): {"image1.jpg": ["caption1", "caption2", ...], ...}
    格式2 (列表): [{"image": "image1.jpg", "captions": ["caption1", ...]}, ...]
    格式3 (COCO格式): {"images": [...], "annotations": [...]}
    格式4 (RSICD官方): {"images": [{"filename":..., "sentences": [{"raw":...}], "split":...}], ...}
    格式5 (OPMMD): {"data": [{"filename":..., "sentences": [{"raw":...}]}], "metadata": {...}}
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import open_clip
from open_clip.cross_modal import compute_cross_modal_similarity


# ============================================================================
# RetrievalDataset (支持 RSICD / OPMMD 等多种格式)
# ============================================================================

class RetrievalDataset(Dataset):
    """
    图文检索数据集（支持 RSICD 和 OPMMD 格式）。
    每张图片可能对应多个 caption，每个 (image, caption) 对展开为独立样本。
    """

    def __init__(
        self,
        image_root: str,
        caption_file: str,
        transform=None,
    ):
        """
        Args:
            image_root: 图片根目录路径
            caption_file: 标注 JSON 文件路径
            transform: 图像预处理 transform
        """
        self.transform = transform

        # 图片目录
        img_root_path = Path(image_root)
        if not img_root_path.exists():
            raise FileNotFoundError(f"图片目录不存在: {img_root_path}")
        self.image_dir = img_root_path

        # 标注文件
        cap_path = Path(caption_file)
        if not cap_path.exists():
            raise FileNotFoundError(f"标注文件不存在: {cap_path}")

        with open(cap_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.samples = self._parse_annotations(raw_data)

        # 构建图片名 -> 样本索引列表
        self.image_to_indices: Dict[str, List[int]] = {}
        self._unique_images: List[str] = []
        for idx, s in enumerate(self.samples):
            img_name = s["image"]
            if img_name not in self.image_to_indices:
                self.image_to_indices[img_name] = []
                self._unique_images.append(img_name)
            self.image_to_indices[img_name].append(idx)

        self.image_to_uid: Dict[str, int] = {
            name: i for i, name in enumerate(self._unique_images)
        }

    @staticmethod
    def _is_rsicd_format(data: dict) -> bool:
        if not isinstance(data, dict):
            return False
        if "images" not in data:
            return False
        if "annotations" in data:
            return False
        imgs = data.get("images")
        if isinstance(imgs, list) and len(imgs) > 0:
            first = imgs[0]
            if isinstance(first, dict) and "sentences" in first:
                return True
        return False

    @staticmethod
    def _is_opmmd_format(data: dict) -> bool:
        """检测 OPMMD JSON 格式: {"data": [...], "metadata": {...}}"""
        if not isinstance(data, dict):
            return False
        if "data" not in data:
            return False
        items = data.get("data")
        if isinstance(items, list) and len(items) > 0:
            first = items[0]
            if isinstance(first, dict) and "filename" in first and "sentences" in first:
                return True
        return False

    def _parse_annotations(self, raw_data) -> List[Dict]:
        samples = []

        # 格式5: OPMMD 格式
        if self._is_opmmd_format(raw_data):
            for item in raw_data["data"]:
                img_name = item.get("filename") or item.get("image") or item.get("file_name")
                if not img_name.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp')):
                    img_name = img_name + '.tif'
                sentences = item.get("sentences", [])
                for sent in sentences:
                    cap = sent.get("raw") or sent.get("caption") or ""
                    if cap:
                        samples.append({"image": img_name, "caption": cap.strip()})
            print(f"[OPMMD] 解析格式: OPMMD, 样本数={len(samples)}")
            return samples

        # 格式4: RSICD 官方格式
        if self._is_rsicd_format(raw_data):
            for item in raw_data["images"]:
                img_name = item.get("filename") or item.get("image") or item.get("file_name")
                sentences = item.get("sentences", [])
                for sent in sentences:
                    cap = sent.get("raw") or sent.get("caption") or ""
                    if cap:
                        samples.append({"image": img_name, "caption": cap.strip()})
            print(f"[RSICD] 解析格式: RSICD官方, 样本数={len(samples)}")
            return samples

        # 格式1: 简单字典
        if isinstance(raw_data, dict):
            keys = list(raw_data.keys())
            if keys and any("." in str(k) for k in keys[:5]):
                for img_name, captions in raw_data.items():
                    if isinstance(captions, str):
                        captions = [captions]
                    for cap in captions:
                        samples.append({"image": img_name, "caption": cap})
                print(f"[简单字典] 解析格式, 样本数={len(samples)}")
                return samples

        # 格式2: 列表
        if isinstance(raw_data, list):
            for item in raw_data:
                img_name = item.get("image") or item.get("filename") or item.get("img_name")
                captions = item.get("captions") or item.get("caption") or []
                if isinstance(captions, str):
                    captions = [captions]
                for cap in captions:
                    samples.append({"image": img_name, "caption": cap})
            print(f"[列表] 解析格式, 样本数={len(samples)}")
            return samples

        # 格式3: COCO
        if isinstance(raw_data, dict) and "images" in raw_data and "annotations" in raw_data:
            id_to_img = {img["id"]: img["file_name"] for img in raw_data["images"]}
            for ann in raw_data["annotations"]:
                img_id = ann["image_id"]
                img_name = id_to_img.get(img_id, str(img_id) + ".jpg")
                samples.append({"image": img_name, "caption": ann["caption"]})
            print(f"[COCO] 解析格式, 样本数={len(samples)}")
            return samples

        raise ValueError(f"无法识别的 JSON 格式，请检查标注文件")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = self._resolve_image_path(sample["image"])

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "caption": sample["caption"],
            "img_name": sample["image"],
            "idx": idx,
        }

    def _resolve_image_path(self, img_name: str) -> Path:
        img_path = self.image_dir / img_name
        if img_path.exists():
            return img_path
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            candidate = img_path.with_suffix(ext)
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"图片不存在: {img_path}")

    @property
    def unique_images(self) -> List[str]:
        return self._unique_images

    def get_captions_for_image(self, img_name: str) -> List[str]:
        indices = self.image_to_indices.get(img_name, [])
        return [self.samples[i]["caption"] for i in indices]

    def get_all_captions(self) -> List[str]:
        return [s["caption"] for s in self.samples]

    def get_image_id_for_caption(self, caption_idx: int) -> int:
        img_name = self.samples[caption_idx]["image"]
        return self.image_to_uid[img_name]


# ============================================================================
# Feature Extraction
# ============================================================================

@torch.no_grad()
def infer_embed_dim(model, tokenizer, device: torch.device) -> int:
    tokens = tokenizer(["test"]).to(device)
    feat = model.encode_text(tokens)
    return feat.shape[-1]


@torch.no_grad()
def extract_unique_image_features(
    model,
    dataset: RetrievalDataset,
    batch_size: int,
    device: torch.device,
    embed_dim: int,
    num_workers: int = 4,
    return_tokens: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    model.eval()
    unique_images = dataset.unique_images
    image_to_uid = dataset.image_to_uid
    u = len(unique_images)

    class UniqueImageDataset(Dataset):
        def __init__(self, ds, names, transform):
            self.ds = ds
            self.names = names
            self.transform = transform

        def __len__(self):
            return len(self.names)

        def __getitem__(self, idx):
            name = self.names[idx]
            path = self.ds._resolve_image_path(name)
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, name

    unique_ds = UniqueImageDataset(dataset, unique_images, dataset.transform)
    loader = DataLoader(
        unique_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    if return_tokens:
        test_img = torch.randn(1, 3, model.visual.image_size[0], model.visual.image_size[1], device=device)
        _, test_tokens = model.encode_image(test_img, return_tokens=True)
        num_patches = test_tokens.shape[1]
        token_dim = test_tokens.shape[2]
        features = torch.zeros(u, embed_dim, device=device)
        all_tokens = torch.zeros(u, num_patches, token_dim, device=device)
    else:
        features = torch.zeros(u, embed_dim, device=device)

    print(f"[Image] 唯一图片: {u}, batch_size={batch_size}, num_workers={num_workers}")
    for images, img_names in tqdm(loader, desc="Image Features"):
        images = images.to(device, non_blocking=True)
        result = model.encode_image(images, return_tokens=return_tokens)
        if return_tokens:
            feats, tokens = result
        else:
            feats = result
        feats = feats / feats.norm(dim=-1, keepdim=True)

        for i, name in enumerate(img_names):
            uid = image_to_uid[name]
            features[uid] = feats[i]
            if return_tokens:
                all_tokens[uid] = tokens[i]

    if return_tokens:
        return features, all_tokens
    return features


@torch.no_grad()
def extract_all_text_features(
    model,
    dataset: RetrievalDataset,
    tokenizer,
    batch_size: int,
    device: torch.device,
    embed_dim: int,
    return_tokens: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    model.eval()
    all_captions = dataset.get_all_captions()
    m = len(all_captions)

    if return_tokens:
        test_tokens = tokenizer(["test"]).to(device)
        _, test_tok = model.encode_text(test_tokens, return_tokens=True)
        num_txt_tokens = test_tok.shape[1]
        token_dim = test_tok.shape[2]
        features = torch.zeros(m, embed_dim, device=device)
        all_tokens = torch.zeros(m, num_txt_tokens, token_dim, device=device)
    else:
        features = torch.zeros(m, embed_dim, device=device)

    print(f"[Text] captions: {m}, batch_size={batch_size}")
    for i in tqdm(range(0, m, batch_size), desc="Text Features"):
        batch = all_captions[i : i + batch_size]
        tokens = tokenizer(batch).to(device, non_blocking=True)
        result = model.encode_text(tokens, return_tokens=return_tokens)
        if return_tokens:
            feats, tok_embs = result
            all_tokens[i : i + batch_size] = tok_embs
        else:
            feats = result
        feats = feats / feats.norm(dim=-1, keepdim=True)
        features[i : i + batch_size] = feats

    if return_tokens:
        return features, all_tokens
    return features


# ============================================================================
# Recall@K Evaluation
# ============================================================================

def compute_recall_from_similarity(
    similarity: torch.Tensor,
    dataset: RetrievalDataset,
    ks: List[int] = (1, 5, 10),
) -> Dict[str, float]:
    device = similarity.device
    u = similarity.shape[0]
    m = similarity.shape[1]

    caption_to_uid = torch.zeros(m, dtype=torch.long, device=device)
    for j in range(m):
        caption_to_uid[j] = dataset.get_image_id_for_caption(j)

    i2t = {}
    for k in ks:
        _, topk = torch.topk(similarity, k=min(k, m), dim=1)
        correct = 0
        for u_idx in range(u):
            candidate_captions = topk[u_idx]
            if (caption_to_uid[candidate_captions] == u_idx).any():
                correct += 1
        i2t[f"I2T_R@{k}"] = correct / u * 100.0

    t2i = {}
    for k in ks:
        _, topk = torch.topk(similarity.T, k=min(k, u), dim=1)
        correct = 0
        for j in range(m):
            target_uid = caption_to_uid[j]
            if target_uid in topk[j]:
                correct += 1
        t2i[f"T2I_R@{k}"] = correct / m * 100.0

    metrics = {}
    metrics.update(i2t)
    metrics.update(t2i)

    mean_recalls = []
    for k in ks:
        mean_rk = (i2t[f"I2T_R@{k}"] + t2i[f"T2I_R@{k}"]) / 2.0
        metrics[f"Mean_R@{k}"] = mean_rk
        mean_recalls.append(mean_rk)

    metrics["Mean_Recall"] = sum(mean_recalls) / len(mean_recalls)
    return metrics


def compute_recall(
    model,
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    dataset: RetrievalDataset,
    ks: List[int] = (1, 5, 10),
) -> Dict[str, float]:
    logit_scale = model.logit_scale.exp() if hasattr(model, 'logit_scale') else 1.0
    similarity = logit_scale * (image_features @ text_features.T)
    return compute_recall_from_similarity(similarity, dataset, ks)


# ============================================================================
# Main (可独立运行)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="OPMMD/RSICD Image-Text Retrieval (OpenCLIP)"
    )
    parser.add_argument("--model", type=str, default="ViT-B-16")
    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--image_root", type=str, default=None,
                        help="图片根目录 (OPMMD 格式, 如 /root/autodl-tmp/OPMMD/image/test)")
    parser.add_argument("--caption_file", type=str, required=True,
                        help="标注 JSON 文件路径")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--output", type=str, default=None,
                        help="输出结果 JSON 路径")
    parser.add_argument("--use_cross_modal", action="store_true", default=False)
    parser.add_argument("--fusion_alpha", type=float, default=0.5)
    parser.add_argument("--cross_heads", type=int, default=8)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("  OPMMD/RSICD Image-Text Retrieval")
    print(f"  Model:       {args.model}")
    print(f"  Pretrained:  {args.pretrained}")
    print(f"  Image Root:  {args.image_root}")
    print(f"  Caption File:{args.caption_file}")
    print(f"  Device:      {device}")
    print("=" * 70)

    # 1. 加载模型
    print("\n[1/5] 加载 OpenCLIP 模型 ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device,
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model)
    embed_dim = infer_embed_dim(model, tokenizer, device)
    print(f"  embed_dim={embed_dim}")

    if args.use_cross_modal:
        print(f"\n[跨模态] 启用, fusion_alpha={args.fusion_alpha}")
        if hasattr(model, 'visual') and hasattr(model.visual, 'output_tokens'):
            model.visual.output_tokens = True

    # 2. 加载数据集
    print("\n[2/5] 加载数据集 ...")
    full_dataset = RetrievalDataset(
        image_root=args.image_root,
        caption_file=args.caption_file,
        transform=preprocess,
    )
    all_unique = full_dataset.unique_images
    print(f"  唯一图片: {len(all_unique)}, captions: {len(full_dataset)}")
    print(f"  平均每图 caption: {len(full_dataset) / len(all_unique):.1f}")

    # OPMMD 格式：全部数据作为评估集（无 split 字段）
    eval_dataset = full_dataset
    u_eval = len(eval_dataset.unique_images)
    m_eval = len(eval_dataset)
    print(f"  评估集: {u_eval} 张图片, {m_eval} 条 caption")

    # 3. 提取特征
    print("\n[3/5] 提取特征 ...")
    return_tokens = args.use_cross_modal

    if return_tokens:
        image_features, image_tokens = extract_unique_image_features(
            model, eval_dataset, args.batch_size, device, embed_dim,
            num_workers=args.num_workers, return_tokens=True,
        )
        text_features, text_tokens = extract_all_text_features(
            model, eval_dataset, tokenizer, args.batch_size, device, embed_dim,
            return_tokens=True,
        )
    else:
        image_features = extract_unique_image_features(
            model, eval_dataset, args.batch_size, device, embed_dim,
            num_workers=args.num_workers,
        )
        text_features = extract_all_text_features(
            model, eval_dataset, tokenizer, args.batch_size, device, embed_dim,
        )

    print(f"  image_features: {image_features.shape}")
    print(f"  text_features:  {text_features.shape}")

    # 4. 计算 Recall
    print("\n[4/5] 计算 Recall@K ...")
    if args.use_cross_modal:
        logit_scale = model.logit_scale.exp() if hasattr(model, 'logit_scale') else 1.0
        image_patches = image_tokens[:, 1:, :]
        similarity = compute_cross_modal_similarity(
            image_features=image_features, text_features=text_features,
            image_tokens=image_patches, text_tokens=text_tokens,
            logit_scale=logit_scale, heads=args.cross_heads,
            fusion_alpha=args.fusion_alpha,
        )
        metrics = compute_recall_from_similarity(similarity, eval_dataset, ks=args.ks)
    else:
        metrics = compute_recall(model, image_features, text_features, eval_dataset, ks=args.ks)

    # 5. 输出结果
    print("\n" + "=" * 70)
    print(f"  Recall@K 结果 ({u_eval} images, {m_eval} captions)")
    print("=" * 70)
    for k in args.ks:
        print(f"  I2T R@{k}:  {metrics[f'I2T_R@{k}']:.2f}%")
        print(f"  T2I R@{k}:  {metrics[f'T2I_R@{k}']:.2f}%")
    print(f"  Mean Recall: {metrics['Mean_Recall']:.2f}%")

    if args.output:
        output_data = {
            "config": {
                "model": args.model,
                "pretrained": args.pretrained,
                "image_root": args.image_root,
                "caption_file": args.caption_file,
                "num_images": u_eval,
                "num_captions": m_eval,
            },
            "results": {k: round(v, 4) for k, v in metrics.items()},
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"  结果已保存至: {args.output}")

    print("=" * 70)
    print("  完成!")


if __name__ == "__main__":
    main()