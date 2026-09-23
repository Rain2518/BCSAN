"""
训练跨模态匹配头 (CrossModalScorer) - OPMMD 版本
================================================

训练策略:
    - 冻结 OpenCLIP 全部参数（Vision Encoder + Text Encoder）
    - 只训练 CrossModalScorer（投影层 + Cross Attention + MLP 分类头）
    - 损失函数: Multi-Positive InfoNCE + Hard Negative Contrastive + KL 蒸馏
    - 默认在 val 集上评估，按 val_loss 保存最佳模型
    - 图片数据集: OPMMD (tif 格式)
    - 文本数据集: /root/autodl-tmp/dataset/ (JSON 格式)

用法:
    python train_cross_modal_op.py \
        --model ViT-B-16 \
        --pretrained /path/to/open_clip_pytorch_model.bin \
        --image_root /root/autodl-tmp/OPMMD/image \
        --caption_file /root/autodl-tmp/dataset/train.json \
        --val_caption_file /root/autodl-tmp/dataset/val.json \
        --batch_size 32 \
        --epochs 20 \
        --lr 1e-5 \
        --weight_decay 5e-4 \
        --device cuda \
        --multi_positive \
        --hard_neg_weight 0.5 \
        --output checkpoints/cross_modal_opmmd.pth
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# 将 src/ 加入 Python path，确保 open_clip 包可导入
_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import open_clip
from open_clip.cross_modal import CrossModalScorer


# ============================================================================
# OPMMD Dataset (训练用，支持 multi-positive)
# ============================================================================

class OPMMDTrainDataset(Dataset):
    """
    OPMMD 训练/验证数据集。

    图片: OPMMD/image/{split}/ 下的 .tif 文件
    文本标注: dataset/{split}.json (Qwen-VL-Max 生成的描述)

    支持两种模式:
    - single (默认): 每张图片随机采样 1 条 caption
    - multi_positive: 返回所有 captions，利用全部标注信息

    验证模式: 每图随机采样 1 条 caption，shuffle=False 保证可复现。
    """

    def __init__(
        self,
        image_root: str,
        caption_file: str,
        transform=None,
        split: str = "train",
        seed: int = 42,
        multi_positive: bool = False,
    ):
        self.image_root = Path(image_root) / split
        self.transform = transform
        self.split = split
        self.multi_positive = multi_positive

        cap_path = Path(caption_file)
        if not cap_path.exists():
            raise FileNotFoundError(f"标注文件不存在: {cap_path}")

        with open(cap_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # 解析所有样本
        all_parsed = self._parse_annotations(raw_data)

        # 按图片分组: image_name → [caption1, caption2, ...]
        self.image_to_captions: Dict[str, List[str]] = {}
        for s in all_parsed:
            img_name = s["image"]
            if img_name not in self.image_to_captions:
                self.image_to_captions[img_name] = []
            self.image_to_captions[img_name].append(s["caption"])

        # 图片列表（用于 __len__ 和索引）
        self.image_names = sorted(self.image_to_captions.keys())

        # 验证图片文件是否存在
        missing = []
        for img_name in self.image_names:
            img_path = self.image_root / img_name
            if not img_path.exists():
                missing.append(img_name)
        if missing:
            print(f"  [警告] {len(missing)} 张图片在 {self.image_root} 中未找到, "
                  f"将从数据集中移除")
            for m in missing:
                del self.image_to_captions[m]
            self.image_names = sorted(self.image_to_captions.keys())

        # 随机数生成器（single 模式用）
        self.rng = np.random.RandomState(seed + hash(split) % 10007)

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
        """解析 OPMMD JSON 格式的标注。"""
        samples = []
        if self._is_opmmd_format(raw_data):
            for item in raw_data["data"]:
                img_name = item.get("filename") or item.get("image") or item.get("file_name")
                # 确保图片名以 .tif 结尾
                if not img_name.lower().endswith(('.tif', '.tiff')):
                    img_name = img_name + '.tif'
                for sent in item.get("sentences", []):
                    cap = sent.get("raw") or sent.get("caption") or ""
                    if cap:
                        samples.append({"image": img_name, "caption": cap.strip()})
            return samples
        raise ValueError(f"无法识别的 JSON 格式，期望 OPMMD 格式 (含 'data' 键)")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        captions = self.image_to_captions[img_name]

        img_path = self.image_root / img_name
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        if self.multi_positive:
            # 返回所有 captions
            return {
                "image": image,
                "captions": captions,
                "img_name": img_name,
            }
        else:
            # 随机选 1 条 caption（原行为）
            cap_idx = self.rng.randint(0, len(captions))
            caption = captions[cap_idx]
            return {
                "image": image,
                "caption": caption,
                "img_name": img_name,
            }


# ============================================================================
# Tokenizer collate
# ============================================================================

def collate_fn(batch, tokenizer):
    """将 batch 中的图像和文本分别堆叠并 tokenize（single caption 模式）。"""
    images = torch.stack([item["image"] for item in batch])
    captions = [item["caption"] for item in batch]
    tokens = tokenizer(captions)
    return {
        "image": images,
        "caption": captions,
        "tokens": tokens,
    }


def collate_fn_multi_positive(batch, tokenizer):
    """
    Multi-positive collate: 处理每张图片有多个 caption 的情况。

    返回:
        "image": (B, 3, H, W) 图像张量
        "captions": List[List[str]] 每张图片的 caption 列表
        "tokens": (total_captions, 77) 所有 caption 的 token 张量
        "cap_counts": List[int] 每张图片的 caption 数量
    """
    images = torch.stack([item["image"] for item in batch])
    captions_list = [item["captions"] for item in batch]  # List[List[str]]
    cap_counts = [len(caps) for caps in captions_list]

    # 展平所有 captions 并 tokenize
    all_captions = []
    for caps in captions_list:
        all_captions.extend(caps)

    tokens = tokenizer(all_captions)  # (total_captions, 77)

    return {
        "image": images,
        "captions": captions_list,
        "tokens": tokens,
        "cap_counts": cap_counts,
    }


# ============================================================================
# 损失函数
# ============================================================================

def multi_positive_infonce(
    cross_scores: torch.Tensor,
    cap_counts: List[int],
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    多正样本 InfoNCE 损失。

    对于每张图片 i，其对应的 cap_counts[i] 个 caption 都是正样本，
    其他图片的所有 caption 都是负样本。

    Args:
        cross_scores: (B, total_captions) 跨模态匹配分数矩阵
        cap_counts: 每张图片的 caption 数量列表
        temperature: 温度缩放因子

    Returns:
        loss_i2t: image-to-text 方向 loss
        loss_t2i: text-to-image 方向 loss
        loss: 平均 loss
    """
    B = len(cap_counts)
    device = cross_scores.device
    total_caps = cross_scores.shape[1]

    # 构建正样本 mask: (B, total_captions)
    pos_mask = torch.zeros(B, total_caps, device=device)
    offset = 0
    for i, count in enumerate(cap_counts):
        pos_mask[i, offset:offset + count] = 1.0
        offset += count

    # --- Image-to-Text 方向 ---
    scores_i2t = cross_scores / temperature  # (B, total_caps)

    scores_max_i2t = scores_i2t.max(dim=1, keepdim=True)[0].detach()
    scores_i2t = scores_i2t - scores_max_i2t

    pos_exp_i2t = (scores_i2t.exp() * pos_mask).sum(dim=1)  # (B,)
    all_exp_i2t = scores_i2t.exp().sum(dim=1)  # (B,)

    valid_i2t = pos_mask.sum(dim=1) > 0  # (B,)
    loss_i2t = torch.zeros(B, device=device)
    loss_i2t[valid_i2t] = -torch.log(
        pos_exp_i2t[valid_i2t] / (all_exp_i2t[valid_i2t] + 1e-8)
    )
    loss_i2t = loss_i2t.mean()

    # --- Text-to-Image 方向 ---
    scores_t2i = cross_scores.T / temperature  # (total_caps, B)

    scores_max_t2i = scores_t2i.max(dim=1, keepdim=True)[0].detach()
    scores_t2i = scores_t2i - scores_max_t2i

    pos_mask_t2i = pos_mask.T  # (total_caps, B)

    pos_exp_t2i = (scores_t2i.exp() * pos_mask_t2i).sum(dim=1)  # (total_caps,)
    all_exp_t2i = scores_t2i.exp().sum(dim=1)  # (total_caps,)

    valid_t2i = pos_mask_t2i.sum(dim=1) > 0  # (total_caps,)
    loss_t2i = torch.zeros(total_caps, device=device)
    loss_t2i[valid_t2i] = -torch.log(
        pos_exp_t2i[valid_t2i] / (all_exp_t2i[valid_t2i] + 1e-8)
    )
    loss_t2i = loss_t2i.mean()

    loss = (loss_i2t + loss_t2i) / 2.0

    return loss, loss_i2t, loss_t2i


def hard_negative_loss(
    cross_scores: torch.Tensor,
    cap_counts: List[int],
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Hard negative contrastive loss.

    对每张图片，找到最相似的错误 caption（hard negative），
    然后计算: -log(exp(pos) / (exp(pos) + exp(hard_neg)))

    Args:
        cross_scores: (B, total_captions) 跨模态匹配分数矩阵
        cap_counts: 每张图片的 caption 数量列表
        temperature: 温度缩放因子

    Returns:
        hard_loss: 平均 hard negative loss
    """
    B = len(cap_counts)
    device = cross_scores.device
    total_caps = cross_scores.shape[1]

    pos_mask = torch.zeros(B, total_caps, device=device)
    offset = 0
    for i, count in enumerate(cap_counts):
        pos_mask[i, offset:offset + count] = 1.0
        offset += count

    neg_mask = 1.0 - pos_mask

    scores = cross_scores / temperature  # (B, total_caps)

    scores_neg = scores.clone()
    scores_neg = scores_neg.masked_fill(pos_mask.bool(), -1e9)
    hard_neg_scores = scores_neg.max(dim=1)[0]  # (B,)

    scores_pos = scores.clone()
    scores_pos = scores_pos.masked_fill(neg_mask.bool(), -1e9)

    pos_exp_sum = scores_pos.exp().sum(dim=1)  # (B,)
    pos_counts = pos_mask.sum(dim=1)  # (B,)
    valid = pos_counts > 0
    pos_score = torch.zeros(B, device=device)
    pos_score[valid] = torch.log(pos_exp_sum[valid] / pos_counts[valid])

    hard_loss = torch.zeros(B, device=device)
    hard_loss[valid] = -pos_score[valid] + torch.log(
        torch.exp(pos_score[valid]) + torch.exp(hard_neg_scores[valid]) + 1e-8
    )

    return hard_loss.mean()


# ============================================================================
# 验证集评估
# ============================================================================

@torch.no_grad()
def validate(
    model,
    cross_scorer: CrossModalScorer,
    val_loader: DataLoader,
    device: torch.device,
    clip_loss_weight: float = 0.5,
    logit_scale: float = 100.0,
    multi_positive: bool = False,
) -> Dict[str, float]:
    """
    在验证集上计算联合 InfoNCE loss (CLIP + CrossModal)。

    当 multi_positive=True 时，使用多正样本版本的 InfoNCE，
    让训练/验证目标保持一致。
    """
    cross_scorer.eval()
    total_loss = 0.0
    total_loss_i2t = 0.0
    total_loss_t2i = 0.0
    total_cross_loss = 0.0
    total_clip_loss = 0.0
    num_batches = 0

    for batch in val_loader:
        images = batch["image"].to(device, non_blocking=True)
        tokens = batch["tokens"].to(device, non_blocking=True)

        B = images.shape[0]

        img_feat, img_tokens = model.encode_image(images, return_tokens=True)
        img_patches = img_tokens[:, 1:, :]  # (B, N_patches, img_dim)

        if multi_positive:
            cap_counts = batch["cap_counts"]
            total_caps = tokens.shape[0]

            txt_feat, txt_tokens = model.encode_text(tokens, return_tokens=True)

            img_norm = F.normalize(img_feat, p=2, dim=-1)
            txt_norm = F.normalize(txt_feat, p=2, dim=-1)
            clip_scores_full = logit_scale * (img_norm @ txt_norm.T)  # (B, total_caps)

            clip_loss, clip_loss_i2t, clip_loss_t2i = multi_positive_infonce(
                clip_scores_full,
                cap_counts,
                temperature=1.0,
            )

            cross_scores = compute_cross_scores_multi_positive(
                cross_scorer,
                img_patches,
                txt_tokens,
            )
            cross_loss, cross_loss_i2t, cross_loss_t2i = multi_positive_infonce(
                cross_scores,
                cap_counts,
                temperature=1.0,
            )
        else:
            txt_feat, txt_tokens = model.encode_text(tokens, return_tokens=True)

            # --- 1. CLIP 原始相似度 (global embedding) ---
            img_norm = F.normalize(img_feat, p=2, dim=-1)
            txt_norm = F.normalize(txt_feat, p=2, dim=-1)
            clip_scores = logit_scale * (img_norm @ txt_norm.T)  # (B, B)

            labels = torch.arange(B, device=device)
            clip_loss_i2t = F.cross_entropy(clip_scores, labels)
            clip_loss_t2i = F.cross_entropy(clip_scores.T, labels)
            clip_loss = (clip_loss_i2t + clip_loss_t2i) / 2.0

            # --- 2. CrossModal 匹配分数 (cosine similarity, token-level) ---
            cross_scores = cross_scorer.forward_pairwise_similarity(
                image_tokens=img_patches,
                text_tokens=txt_tokens,
            )  # (B, B)

            cross_loss_i2t = F.cross_entropy(cross_scores, labels)
            cross_loss_t2i = F.cross_entropy(cross_scores.T, labels)
            cross_loss = (cross_loss_i2t + cross_loss_t2i) / 2.0

        # --- 3. 联合损失 ---
        loss = (1.0 - clip_loss_weight) * cross_loss + clip_loss_weight * clip_loss

        if multi_positive:
            loss_i2t_item = (1.0 - clip_loss_weight) * cross_loss_i2t.item() + clip_loss_weight * clip_loss_i2t.item()
            loss_t2i_item = (1.0 - clip_loss_weight) * cross_loss_t2i.item() + clip_loss_weight * clip_loss_t2i.item()
        else:
            loss_i2t_item = (1.0 - clip_loss_weight) * cross_loss_i2t.item() + clip_loss_weight * clip_loss_i2t.item()
            loss_t2i_item = (1.0 - clip_loss_weight) * cross_loss_t2i.item() + clip_loss_weight * clip_loss_t2i.item()

        total_loss += loss.item()
        total_loss_i2t += loss_i2t_item
        total_loss_t2i += loss_t2i_item
        total_cross_loss += cross_loss.item()
        total_clip_loss += clip_loss.item()
        num_batches += 1

    n = max(num_batches, 1)
    return {
        "loss": total_loss / n,
        "loss_i2t": total_loss_i2t / n,
        "loss_t2i": total_loss_t2i / n,
        "cross_loss": total_cross_loss / n,
        "clip_loss": total_clip_loss / n,
    }


# ============================================================================
# 辅助函数：在训练中计算 B×(B*num_caps) 跨模态分数矩阵
# ============================================================================

def compute_cross_scores_multi_positive(
    cross_scorer: CrossModalScorer,
    img_patches: torch.Tensor,      # (B, N_img, img_dim)
    txt_tokens: torch.Tensor,       # (total_caps, N_txt, txt_dim)
    chunk_size: int = 32,
) -> torch.Tensor:
    """
    计算 B 张图像与 total_captions 条文本的跨模态 cosine similarity 矩阵。

    使用 cross_scorer._compute_cosine_similarity() 逐行计算。
    先预精炼两侧特征以减少重复计算。

    Args:
        cross_scorer: CrossModalScorer 实例
        img_patches: (B, N_img, img_dim) 图像 patch tokens
        txt_tokens: (total_caps, N_txt, txt_dim) 文本 tokens
        chunk_size: 每次处理的图像行数

    Returns:
        scores: (B, total_caps) cosine similarity 矩阵 (已缩放)
    """
    B = img_patches.shape[0]
    total_caps = txt_tokens.shape[0]
    device = img_patches.device

    img_refined = cross_scorer.refine_image(img_patches)     # (B, N_img, img_dim)
    txt_refined = cross_scorer.refine_text(txt_tokens)       # (total_caps, N_txt, txt_dim)

    scores = torch.zeros(B, total_caps, device=device)

    for i in range(B):
        img_i = img_refined[i:i+1].expand(total_caps, -1, -1)  # (total_caps, N_img, img_dim)
        s = cross_scorer._compute_cosine_similarity(img_i, txt_refined)  # (total_caps, 1)
        scores[i, :] = s.squeeze(-1)

    return scores


# ============================================================================
# Training
# ============================================================================

def train():
    parser = argparse.ArgumentParser(
        description="Train CrossModalScorer on OPMMD"
    )

    parser.add_argument("--model", type=str, default="ViT-B-16")
    parser.add_argument("--pretrained", type=str, required=True,
                        help="预训练模型路径 (如 /root/autodl-tmp/ViT-B-32.pt)")
    parser.add_argument("--image_root", type=str, default="/root/autodl-tmp/OPMMD/image",
                        help="OPMMD 图片根目录，其下有 train/val/test 子目录")
    parser.add_argument("--caption_file", type=str, default="/root/autodl-tmp/dataset/train.json",
                        help="训练集文本标注 JSON 文件路径")
    parser.add_argument("--val_caption_file", type=str, default="/root/autodl-tmp/dataset/val.json",
                        help="验证集文本标注 JSON 文件路径")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="学习率 (降低防止过拟合，建议 1e-5 ~ 2e-5)")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="权重衰减 (增大防止过拟合，建议 5e-4 ~ 1e-3)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output", type=str, default="checkpoints/cross_modal_opmmd.pth")
    parser.add_argument("--cross_heads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_val", action="store_true", default=False,
                        help="不使用验证集，仅按训练 loss 保存 (调试用)")
    parser.add_argument("--save_every", type=int, default=0,
                        help="每隔 N 个 epoch 保存一次 checkpoint (0=不启用, 仅保存最佳)")
    parser.add_argument("--kl_weight", type=float, default=0.1,
                        help="KL 蒸馏损失权重 (将 CLIP 匹配分布作为 teacher 约束 CrossModal, 默认 0.1)")
    parser.add_argument("--clip_loss_weight", type=float, default=0.7,
                        help="[已废弃] 请使用 --kl_weight")
    parser.add_argument("--enable_per_side_refiner", action="store_true", default=True,
                        help="启用 PerSideRefiner (默认开启, 模态内语义增强)")
    parser.add_argument("--no_enable_per_side_refiner", action="store_false", dest="enable_per_side_refiner",
                        help="关闭 PerSideRefiner")
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                        help="Early stopping 耐心值 (连续 N 个 epoch val_loss 不下降即停止, 默认 5)")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="CrossModalScorer 中的 dropout 概率 (默认 0.2)")
    parser.add_argument("--multi_positive", action="store_true", default=False,
                        help="启用 multi-positive 训练：每个 image 使用全部 caption 作为正样本")
    parser.add_argument("--hard_neg_weight", type=float, default=0.0,
                        help="Hard negative contrastive loss 权重 (默认 0.0 禁用)")
    parser.add_argument("--mp_temperature", type=float, default=1.0,
                        help="Multi-positive InfoNCE 的温度参数 (默认 1.0)")

    # ---- 结构消融开关 ----
    parser.add_argument("--no_use_cross_attention", action="store_false",
                        dest="use_cross_attention", default=True,
                        help="消融: 关闭 Bidirectional Cross Attention (w/o BCA)")
    parser.add_argument("--no_use_token_similarity", action="store_false",
                        dest="use_token_similarity", default=True,
                        help="消融: 关闭 Token-level Similarity (w/o TLS)")
    parser.add_argument("--no_use_attention_pool", action="store_false",
                        dest="use_attention_pool", default=True,
                        help="消融: 关闭 Attention Pooling, 回退为 mean pooling (w/o AP)")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"数据集: OPMMD (图片) + dataset/ (文本)")
    print(f"配置: model={args.model}, epochs={args.epochs}, lr={args.lr}, "
          f"batch_size={args.batch_size}, cross_heads={args.cross_heads}")
    print(f"图片根目录: {args.image_root}")
    print(f"训练标注: {args.caption_file}")
    print(f"验证标注: {args.val_caption_file}")
    print(f"验证集: {'禁用' if args.no_val else '启用'}")
    print(f"PerSideRefiner: {'启用' if args.enable_per_side_refiner else '关闭'}")
    print(f"Multi-Positive: {'启用' if args.multi_positive else '关闭'}")
    print(f"Hard Neg Weight: {args.hard_neg_weight}")
    print(f"损失函数: {'Multi-Positive InfoNCE' if args.multi_positive else 'InfoNCE'} "
          f"+ {args.kl_weight} * KL(CrossModal || CLIP)"
          f"{' + ' + str(args.hard_neg_weight) + ' * HardNegLoss' if args.hard_neg_weight > 0 else ''}"
          f", dropout={args.dropout}")

    # 打印消融配置
    ablation_parts = []
    if args.use_cross_attention:
        ablation_parts.append("BCA")
    if args.use_token_similarity:
        ablation_parts.append("TLS")
    if args.use_attention_pool:
        ablation_parts.append("AP")
    if not ablation_parts:
        ablation_parts.append("Baseline (projection+cosine only)")
    print(f"结构消融: {' + '.join(ablation_parts)}")
    print(f"  BCA(Bidirectional Cross Attention): {'✓' if args.use_cross_attention else '✗'}")
    print(f"  TLS(Token-level Similarity):        {'✓' if args.use_token_similarity else '✗'}")
    print(f"  AP(Attention Pooling):             {'✓' if args.use_attention_pool else '✗'}")

    # ============================================================
    # 1. 加载 OpenCLIP (冻结)
    # ============================================================
    print("\n[1/5] 加载 OpenCLIP 模型 ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model,
        pretrained=args.pretrained,
        device=device,
    )
    model.eval()

    # 冻结所有 CLIP 参数
    for p in model.parameters():
        p.requires_grad = False

    tokenizer = open_clip.get_tokenizer(args.model)

    # 开启 vision 的 token 输出
    if hasattr(model, 'visual') and hasattr(model.visual, 'output_tokens'):
        model.visual.output_tokens = True

    # 推断维度 & CLIP logit_scale
    with torch.no_grad():
        test_tokens = tokenizer(["test"]).to(device)
        feat = model.encode_text(test_tokens)
        embed_dim = feat.shape[-1]

        test_img = torch.randn(1, 3, model.visual.image_size[0], model.visual.image_size[1], device=device)
        _, img_tok = model.encode_image(test_img, return_tokens=True)
        img_token_dim = img_tok.shape[-1]
        _, txt_tok = model.encode_text(test_tokens, return_tokens=True)
        txt_token_dim = txt_tok.shape[-1]

        if hasattr(model, 'logit_scale'):
            logit_scale = model.logit_scale.exp().item()
        else:
            logit_scale = 100.0

    print(f"  embed_dim={embed_dim}, img_token_dim={img_token_dim}, txt_token_dim={txt_token_dim}")
    print(f"  CLIP logit_scale={logit_scale:.2f}")

    # ============================================================
    # 2. 创建 CrossModalScorer (可训练)
    # ============================================================
    print("\n[2/5] 创建 CrossModalScorer ...")
    cross_scorer = CrossModalScorer(
        img_dim=img_token_dim,
        txt_dim=txt_token_dim,
        heads=args.cross_heads,
        dropout=args.dropout,
        enable_per_side_refiner=args.enable_per_side_refiner,
        use_cross_attention=args.use_cross_attention,
        use_token_similarity=args.use_token_similarity,
        use_attention_pool=args.use_attention_pool,
    ).to(device)

    total_params = sum(p.numel() for p in cross_scorer.parameters())
    trainable_params = sum(p.numel() for p in cross_scorer.parameters() if p.requires_grad)
    print(f"  CrossModalScorer 参数量: {total_params:,} (可训练: {trainable_params:,})")

    # ============================================================
    # 3. 加载数据 (train + val)
    # ============================================================
    print("\n[3/5] 加载 OPMMD 数据 ...")

    from functools import partial

    # --- 训练集 ---
    train_dataset = OPMMDTrainDataset(
        image_root=args.image_root,
        caption_file=args.caption_file,
        transform=preprocess,
        split="train",
        seed=args.seed,
        multi_positive=args.multi_positive,
    )

    if args.multi_positive:
        cap_counts = [len(caps) for caps in train_dataset.image_to_captions.values()]
        avg_caps = sum(cap_counts) / len(cap_counts) if cap_counts else 0
        print(f"  训练集: {len(train_dataset)} 张图片 "
              f"(multi-positive, 平均 {avg_caps:.1f} captions/图, 共 {sum(cap_counts)} captions)")
        train_collate = partial(collate_fn_multi_positive, tokenizer=tokenizer)
    else:
        print(f"  训练集: {len(train_dataset)} 张图片 "
              f"(每 epoch 随机采 1 caption/图)")
        train_collate = partial(collate_fn, tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=train_collate,
        drop_last=True,
    )

    # --- 验证集 ---
    val_loader = None
    if not args.no_val:
        try:
            val_dataset = OPMMDTrainDataset(
                image_root=args.image_root,
                caption_file=args.val_caption_file,
                transform=preprocess,
                split="val",
                seed=args.seed,
                multi_positive=args.multi_positive,
            )
            if args.multi_positive:
                val_collate = partial(collate_fn_multi_positive, tokenizer=tokenizer)
                print(f"  验证集: {len(val_dataset)} 张图片 (multi-positive, 使用全部 caption)")
            else:
                val_collate = partial(collate_fn, tokenizer=tokenizer)
                print(f"  验证集: {len(val_dataset)} 张图片 (single caption, 固定 seed 可复现)")

            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
                collate_fn=val_collate,
            )
        except (ValueError, FileNotFoundError) as e:
            print(f"  [警告] 无法加载验证集: {e}")
            print(f"  [警告] 回退到仅按训练 loss 保存模型")
            val_loader = None

    # ============================================================
    # 4. 训练循环
    # ============================================================
    print(f"\n[4/5] 开始训练 (epochs={args.epochs}, lr={args.lr}) ...\n")

    optimizer = torch.optim.AdamW(
        cross_scorer.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_config = {
        "exp_tag": "OPMMD",
        "dataset": "OPMMD",
        "model": args.model,
        "pretrained": args.pretrained,
        "img_token_dim": img_token_dim,
        "txt_token_dim": txt_token_dim,
        "cross_heads": args.cross_heads,
        "bidirectional": True,
        "enable_per_side_refiner": args.enable_per_side_refiner,
        "use_cross_attention": args.use_cross_attention,
        "use_token_similarity": args.use_token_similarity,
        "use_attention_pool": args.use_attention_pool,
        "clip_loss_weight": args.clip_loss_weight,
        "multi_positive": args.multi_positive,
        "hard_neg_weight": args.hard_neg_weight,
        "mp_temperature": args.mp_temperature,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }

    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_epoch = 0
    early_stop_counter = 0
    history: List[Dict] = []

    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        # ---- 训练 ----
        cross_scorer.train()
        train_loss_sum = 0.0
        train_cross_loss_sum = 0.0
        train_clip_loss_sum = 0.0
        train_hard_loss_sum = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [train]")
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            tokens = batch["tokens"].to(device, non_blocking=True)

            B = images.shape[0]

            # 提取图像特征和 token (冻结，无梯度)
            with torch.no_grad():
                img_feat, img_tokens = model.encode_image(images, return_tokens=True)
            img_patches = img_tokens[:, 1:, :]  # (B, N_patches, img_dim)

            if args.multi_positive:
                # ---- Multi-Positive 训练 ----
                cap_counts = batch["cap_counts"]
                total_caps = tokens.shape[0]

                with torch.no_grad():
                    txt_feat, txt_tokens = model.encode_text(tokens, return_tokens=True)

                # --- 1. CLIP 原始相似度 (teacher, 无梯度) ---
                with torch.no_grad():
                    img_norm = F.normalize(img_feat, p=2, dim=-1)
                    txt_norm = F.normalize(txt_feat, p=2, dim=-1)
                    clip_scores_full = logit_scale * (img_norm @ txt_norm.T)  # (B, total_caps)

                    clip_loss, clip_loss_i2t, clip_loss_t2i = multi_positive_infonce(
                        clip_scores_full, cap_counts,
                        temperature=1.0,
                    )

                # --- 2. CrossModal 匹配分数 (token-level, 有梯度) ---
                cross_scores = compute_cross_scores_multi_positive(
                    cross_scorer, img_patches, txt_tokens,
                )  # (B, total_caps)

                cross_loss, cross_loss_i2t, cross_loss_t2i = multi_positive_infonce(
                    cross_scores, cap_counts,
                    temperature=args.mp_temperature,
                )

                # --- 3. Hard Negative Loss ---
                hard_loss = torch.tensor(0.0, device=device)
                if args.hard_neg_weight > 0:
                    hard_loss = hard_negative_loss(
                        cross_scores, cap_counts,
                        temperature=args.mp_temperature,
                    )

                # --- 4. KL 蒸馏损失 ---
                with torch.no_grad():
                    clip_probs_i2t = F.softmax(clip_scores_full, dim=1).detach()
                    clip_probs_t2i = F.softmax(clip_scores_full.T, dim=1).detach()

                cross_log_probs_i2t = F.log_softmax(cross_scores, dim=1)
                cross_log_probs_t2i = F.log_softmax(cross_scores.T, dim=1)

                kl_loss_i2t = F.kl_div(cross_log_probs_i2t, clip_probs_i2t, reduction='batchmean')
                kl_loss_t2i = F.kl_div(cross_log_probs_t2i, clip_probs_t2i, reduction='batchmean')
                kl_loss = (kl_loss_i2t + kl_loss_t2i) / 2.0

                # --- 5. 总损失 ---
                loss = cross_loss + args.kl_weight * kl_loss + args.hard_neg_weight * hard_loss

                clip_loss_val = clip_loss.item()

            else:
                # ---- 原始 Single-Caption 训练 ----
                with torch.no_grad():
                    txt_feat, txt_tokens = model.encode_text(tokens, return_tokens=True)

                # --- 1. CLIP 原始相似度 (teacher, 无梯度) ---
                with torch.no_grad():
                    img_norm = F.normalize(img_feat, p=2, dim=-1)
                    txt_norm = F.normalize(txt_feat, p=2, dim=-1)
                    clip_scores = logit_scale * (img_norm @ txt_norm.T)  # (B, B)

                    labels = torch.arange(B, device=device)
                    clip_loss_i2t = F.cross_entropy(clip_scores, labels)
                    clip_loss_t2i = F.cross_entropy(clip_scores.T, labels)
                    clip_loss = (clip_loss_i2t + clip_loss_t2i) / 2.0

                # --- 2. CrossModal 匹配分数 (token-level, 有梯度) ---
                cross_scores = cross_scorer.forward_pairwise_similarity(
                    image_tokens=img_patches,
                    text_tokens=txt_tokens,
                )  # (B, B)

                cross_loss_i2t = F.cross_entropy(cross_scores, labels)
                cross_loss_t2i = F.cross_entropy(cross_scores.T, labels)
                cross_loss = (cross_loss_i2t + cross_loss_t2i) / 2.0

                # --- 3. Hard Negative Loss (single-caption) ---
                hard_loss = torch.tensor(0.0, device=device)
                if args.hard_neg_weight > 0:
                    with torch.no_grad():
                        mask = torch.eye(B, device=device).bool()
                        neg_scores = cross_scores.clone()
                        neg_scores.masked_fill_(mask, -1e9)
                        hard_neg_i2t = neg_scores.max(dim=1)[0]
                        hard_neg_t2i = neg_scores.max(dim=0)[0]

                    pos_i2t = cross_scores.diag()
                    pos_t2i = cross_scores.diag()

                    hard_loss_i2t = (-pos_i2t + torch.log(
                        torch.exp(pos_i2t) + torch.exp(hard_neg_i2t) + 1e-8
                    )).mean()
                    hard_loss_t2i = (-pos_t2i + torch.log(
                        torch.exp(pos_t2i) + torch.exp(hard_neg_t2i) + 1e-8
                    )).mean()
                    hard_loss = (hard_loss_i2t + hard_loss_t2i) / 2.0

                # --- 4. KL 蒸馏损失 ---
                with torch.no_grad():
                    clip_probs_i2t = F.softmax(clip_scores, dim=1).detach()
                    clip_probs_t2i = F.softmax(clip_scores.T, dim=1).detach()

                cross_log_probs_i2t = F.log_softmax(cross_scores, dim=1)
                cross_log_probs_t2i = F.log_softmax(cross_scores.T, dim=1)

                kl_loss_i2t = F.kl_div(cross_log_probs_i2t, clip_probs_i2t, reduction='batchmean')
                kl_loss_t2i = F.kl_div(cross_log_probs_t2i, clip_probs_t2i, reduction='batchmean')
                kl_loss = (kl_loss_i2t + kl_loss_t2i) / 2.0

                # --- 5. 总损失 ---
                loss = cross_loss + args.kl_weight * kl_loss + args.hard_neg_weight * hard_loss
                clip_loss_val = clip_loss.item()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_cross_loss_sum += cross_loss.item() if isinstance(cross_loss, torch.Tensor) else cross_loss
            train_clip_loss_sum += clip_loss_val
            train_hard_loss_sum += hard_loss.item() if isinstance(hard_loss, torch.Tensor) else hard_loss
            num_batches += 1

            # 进度条显示
            postfix_dict = {
                "loss": f"{loss.item():.4f}",
                "cross": f"{cross_loss.item() if isinstance(cross_loss, torch.Tensor) else cross_loss:.4f}",
                "kl": f"{kl_loss.item():.4f}",
                "clip": f"{clip_loss_val:.4f}",
            }
            if args.hard_neg_weight > 0:
                postfix_dict["hard"] = f"{hard_loss.item() if isinstance(hard_loss, torch.Tensor) else hard_loss:.4f}"
            pbar.set_postfix(postfix_dict)

        avg_train_loss = train_loss_sum / max(num_batches, 1)
        avg_train_cross = train_cross_loss_sum / max(num_batches, 1)
        avg_train_clip = train_clip_loss_sum / max(num_batches, 1)
        avg_train_hard = train_hard_loss_sum / max(num_batches, 1)

        # ---- 验证 ----
        if val_loader is not None:
            val_metrics = validate(
                model, cross_scorer, val_loader, device,
                clip_loss_weight=args.clip_loss_weight,
                logit_scale=logit_scale,
                multi_positive=args.multi_positive,
            )
            avg_val_loss = val_metrics["loss"]
            current_loss = avg_val_loss
            loss_label = "val_loss"
            hard_str = f", hard={avg_train_hard:.4f}" if args.hard_neg_weight > 0 else ""
            print(f"  Epoch {epoch + 1}: train_loss={avg_train_loss:.4f} "
                  f"(cross={avg_train_cross:.4f}, clip={avg_train_clip:.4f}{hard_str}), "
                  f"val_loss={avg_val_loss:.4f} "
                  f"(cross={val_metrics['cross_loss']:.4f}, clip={val_metrics['clip_loss']:.4f})")
        else:
            avg_val_loss = None
            current_loss = avg_train_loss
            loss_label = "train_loss"
            hard_str = f", hard={avg_train_hard:.4f}" if args.hard_neg_weight > 0 else ""
            print(f"  Epoch {epoch + 1}: train_loss={avg_train_loss:.4f} "
                  f"(cross={avg_train_cross:.4f}, clip={avg_train_clip:.4f}{hard_str})")

        # 记录
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_cross_loss": avg_train_cross,
            "train_clip_loss": avg_train_clip,
        }
        if args.hard_neg_weight > 0:
            epoch_record["train_hard_loss"] = avg_train_hard
        if avg_val_loss is not None:
            epoch_record["val_loss"] = avg_val_loss
            epoch_record["val_cross_loss"] = val_metrics["cross_loss"]
            epoch_record["val_clip_loss"] = val_metrics["clip_loss"]
        history.append(epoch_record)

        # ---- 保存最佳模型 ----
        is_best = current_loss < best_val_loss
        if val_loader is not None:
            best_val_loss = min(best_val_loss, current_loss)
            best_reference = best_val_loss
        else:
            best_reference = best_train_loss
            if current_loss < best_train_loss:
                best_train_loss = current_loss
                is_best = True
            else:
                is_best = False

        if is_best:
            best_epoch = epoch + 1
            early_stop_counter = 0
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": cross_scorer.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": current_loss,
                "loss_type": loss_label,
                "config": train_config,
            }
            torch.save(checkpoint, args.output)
            print(f"  ✓ 保存最佳模型至 {args.output} "
                  f"({loss_label}={current_loss:.4f}, epoch={epoch + 1})")
        else:
            early_stop_counter += 1

        # ---- Early Stopping ----
        if val_loader is not None and early_stop_counter >= args.early_stopping_patience:
            print(f"\n  ⏹ Early stopping: val_loss 连续 {args.early_stopping_patience} 个 epoch 未下降，停止训练")
            print(f"  最佳 epoch: {best_epoch}, 最佳 val_loss: {best_val_loss:.4f}")
            break

        # ---- 定期保存 ----
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(output_dir, f"cross_modal_op_epoch_{epoch + 1}.pth")
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": cross_scorer.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": current_loss,
                "loss_type": loss_label,
                "config": train_config,
            }
            torch.save(checkpoint, ckpt_path)
            print(f"  ✓ 定期保存至 {ckpt_path}")

    # ============================================================
    # 5. 训练完成，输出总结
    # ============================================================
    print(f"\n[5/5] 训练完成!")
    print(f"  最佳 {loss_label}: {best_reference:.4f} (epoch {best_epoch})")
    print(f"  模型已保存至: {args.output}")

    # 保存训练历史
    output_base = os.path.splitext(os.path.basename(args.output))[0]
    history_path = os.path.join(output_dir, f"{output_base}_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": train_config,
            "best_loss": best_reference,
            "best_loss_type": loss_label,
            "best_epoch": best_epoch,
            "history": history,
        }, f, indent=2, ensure_ascii=False)
    print(f"  训练历史已保存至: {history_path}")

    # 打印 loss 曲线概览
    print(f"\n  Loss 曲线:")
    for r in history:
        e = r["epoch"]
        tl = r["train_loss"]
        vl = r.get("val_loss", None)
        tc = r.get("train_cross_loss", None)
        tcl = r.get("train_clip_loss", None)
        th = r.get("train_hard_loss", None)
        marker = " ★" if r["epoch"] == best_epoch else ""
        if vl is not None:
            hard_info = f", hard={th:.4f}" if th is not None else ""
            print(f"    Epoch {e:2d}: train={tl:.4f} (cross={tc:.4f}, clip={tcl:.4f}{hard_info}), "
                  f"val={vl:.4f}{marker}")
        else:
            hard_info = f", hard={th:.4f}" if th is not None else ""
            print(f"    Epoch {e:2d}: train={tl:.4f} (cross={tc:.4f}, clip={tcl:.4f}{hard_info}){marker}")


if __name__ == "__main__":
    train()