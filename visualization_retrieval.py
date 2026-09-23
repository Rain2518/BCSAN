#!/usr/bin/env python
"""
可视化图文检索结果 (Visualization of Image-Text Retrieval)
==========================================================
功能:
    - Image-to-Text: 给定查询图片，展示 Top-K 检索到的 captions
    - Text-to-Image: 给定查询文本，展示 Top-K 检索到的图片
    - 并排对比: CLIP vs BCSAN (最终融合) 检索结果
    - 支持 OPMMD (.tif) 和 RSICD 等数据集

重要说明:
    BCSAN = CLIP + CrossModalScorer 最终融合:
        fused_sim = FUSION_ALPHA * clip_sim + (1 - FUSION_ALPHA) * cross_sim
    可视化展示的是最终融合结果，非 Cross-only。

用法:
    cd GAI-open_clip
    python visualization_retrieval.py

输出:
    retrieval_vis/
    ├── I2T/
    │   ├── CLIP/            # CLIP Image-to-Text
    │   └── BCSAN/           # BCSAN Image-to-Text (融合后)
    ├── T2I/
    │   ├── CLIP/            # CLIP Text-to-Image
    │   └── BCSAN/           # BCSAN Text-to-Image (融合后)
    └── COMPARE/             # CLIP vs BCSAN 并排对比
        ├── compare_I2T_000.png
        └── compare_T2I_000.png
"""

import sys
import os
import json
import textwrap
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# ---- 将 src/ 加入 Python path ----
_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import open_clip
from open_clip.cross_modal import CrossModalScorer

# ---- 导入 retrieval_op 中的工具 ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieval_op import (
    RetrievalDataset,
    extract_unique_image_features,
    extract_all_text_features,
    infer_embed_dim,
)


# ============================================================
# 配置
# ============================================================

MODEL_NAME = "ViT-B-32"
PRETRAINED_PATH = "E:/weights/ViT-B-32.pt"
CROSS_MODAL_CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "checkpoints", "cross_modal_opmmd_fulll.pth")
IMAGE_ROOT = "D:/data/OPMMD/image/test"
CAPTION_FILE = "D:/data/dataset/test.json"
BATCH_SIZE = 64
NUM_WORKERS = 0
DEVICE = "cpu"
TOPK = 5
OUTPUT_DIR = "./retrieval_vis"

# 图片查询索引（用于 I2T 可视化），论文建议选取代表性的 3 个案例
IMAGE_QUERY_INDICES = [0, 5, 10]

# 文本查询索引（用于 T2I 可视化），独立于图片索引，论文建议 3 个案例
TEXT_QUERY_INDICES = [0, 5, 10]

# BCSAN 融合系数: BCSAN = alpha * CLIP + (1-alpha) * CrossModalScorer
# 建议与你实验中 Recall@K 最优的 alpha 保持一致
FUSION_ALPHA = 0.25


# ============================================================
# 工具函数
# ============================================================


def load_image(path):
    """加载图片并转为 RGB。"""
    img = Image.open(path)
    return img.convert("RGB")


def ensure_dir(path):
    """确保目录存在。"""
    os.makedirs(path, exist_ok=True)


# ============================================================
# Image-to-Text 可视化
# ============================================================


def visualize_i2t(
    dataset: RetrievalDataset,
    similarity: torch.Tensor,
    query_idx: int,
    topk: int = TOPK,
    save_path: str = None,
    title_prefix: str = "",
):
    """
    Image-to-Text 检索可视化。

    左侧: 查询图片
    右侧: Top-K 检索到的 captions
          - 绿色 [GREEN] + [GT] = 正确检索 (Ground Truth)
          - 红色 [RED] = 错误检索

    Args:
        dataset:      RetrievalDataset 实例
        similarity:   (U, M) 相似度矩阵
        query_idx:    查询图片的唯一索引 (0..U-1)
        topk:         Top-K
        save_path:    保存路径
        title_prefix: 标题前缀, 如 "CLIP" 或 "BCSAN"
    """
    image_name = dataset.unique_images[query_idx]
    image_path = dataset._resolve_image_path(image_name)
    img = load_image(image_path)

    gt_indices = dataset.image_to_indices.get(image_name, [])
    gt_set = set(gt_indices)

    scores = similarity[query_idx]
    values, indices = torch.topk(scores, min(topk, len(scores)))
    hits = [idx.item() in gt_set for idx in indices]

    top_captions = []
    for rank, (idx, val, hit) in enumerate(zip(indices, values, hits)):
        caption = dataset.samples[idx.item()]["caption"]
        marker = " [GT]" if hit else ""
        top_captions.append((rank + 1, caption, val.item(), marker))

    # ---- 绘图 ----
    fig, (ax_img, ax_txt) = plt.subplots(1, 2, figsize=(16, 7))

    ax_img.imshow(img)
    ax_img.axis("off")
    ax_img.set_title(f"{title_prefix} Query Image\n{image_name}",
                     fontsize=13, fontweight="bold")

    ax_txt.axis("off")
    ax_txt.set_xlim(0, 1)
    ax_txt.set_ylim(0, 1)
    ax_txt.set_title(f"{title_prefix} Top-{topk} Retrieved Captions",
                     fontsize=13, fontweight="bold")

    text_lines = []
    for rank, caption, score, marker in top_captions:
        color = "green" if marker else "red"
        truncated = caption[:120] + ("..." if len(caption) > 120 else "")
        line = (f"#{rank} [{color.upper()}] score={score:.3f}{marker}\n"
                f'   "{truncated}"')
        text_lines.append(line)

    full_text = "\n\n".join(text_lines)
    ax_txt.text(
        0.02, 0.95, full_text,
        fontsize=10, family="monospace",
        verticalalignment="top", transform=ax_txt.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
    )

    plt.tight_layout(pad=2)

    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, f"I2T_{query_idx:03d}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [I2T] 保存至: {save_path}")


# ============================================================
# Text-to-Image 可视化
# ============================================================


def visualize_t2i(
    dataset: RetrievalDataset,
    similarity: torch.Tensor,
    caption_idx: int,
    topk: int = TOPK,
    save_path: str = None,
    title_prefix: str = "",
):
    """
    Text-to-Image 检索可视化。

    左侧: 查询 caption 文本
    右侧: Top-K 检索到的图片
          - 绿色边框 + [GT] = 正确检索
          - 红色边框 = 错误检索

    Args:
        dataset:      RetrievalDataset 实例
        similarity:   (U, M) 相似度矩阵
        caption_idx:  查询 caption 的索引 (0..M-1)
        topk:         Top-K
        save_path:    保存路径
        title_prefix: 标题前缀
    """
    caption = dataset.samples[caption_idx]["caption"]
    gt_uid = dataset.get_image_id_for_caption(caption_idx)

    scores = similarity[:, caption_idx]
    values, indices = torch.topk(scores, min(topk, len(scores)))
    hits = [idx.item() == gt_uid for idx in indices]

    # ---- 绘图 ----
    ncols = topk
    fig, axes = plt.subplots(
        1, ncols + 1, figsize=(4 * (ncols + 1), 4),
        gridspec_kw={"width_ratios": [1.5] + [1] * ncols},
    )

    ax_txt = axes[0]
    ax_txt.axis("off")
    ax_txt.text(
        0.5, 0.5, f'{title_prefix} Query:\n\n"{caption}"',
        fontsize=11, ha="center", va="center",
        transform=ax_txt.transAxes, wrap=True,
        bbox=dict(boxstyle="round,pad=0.8", facecolor="lightyellow", alpha=0.9),
    )
    ax_txt.set_title("Query Text", fontsize=12, fontweight="bold")

    for i, (ax, img_id, score, hit) in enumerate(zip(axes[1:], indices, values, hits)):
        img_name = dataset.unique_images[img_id.item()]
        img_path = dataset._resolve_image_path(img_name)
        img = load_image(img_path)

        ax.imshow(img)
        ax.axis("off")

        border_color = "green" if hit else "red"
        gt_label = " [GT]" if hit else ""
        ax.set_title(
            f"Rank #{i+1}\nscore={score:.3f}{gt_label}",
            fontsize=10,
            fontweight="bold" if hit else "normal",
            color=border_color,
        )
        for spine in ax.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(3 if hit else 1.5)

    plt.suptitle(f"{title_prefix} Text-to-Image Retrieval",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(pad=2)

    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, f"T2I_{caption_idx:03d}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [T2I] 保存至: {save_path}")


# ============================================================
# CLIP vs BCSAN 并排对比 (Image-to-Text)
# ============================================================


def visualize_compare_i2t(
    dataset: RetrievalDataset,
    clip_sim: torch.Tensor,
    bcsan_sim: torch.Tensor,
    query_idx: int,
    topk: int = TOPK,
    save_path: str = None,
):
    """
    Image-to-Text 并排对比: CLIP vs BCSAN.

    上方: 查询图片
    下方左 (蓝色): CLIP Top-K captions
    下方右 (红色): BCSAN Top-K captions

    Args:
        dataset:    RetrievalDataset 实例
        clip_sim:   (U, M) CLIP 相似度矩阵
        bcsan_sim:  (U, M) BCSAN 融合相似度矩阵
                    = FUSION_ALPHA * clip_sim + (1-FUSION_ALPHA) * cross_sim
        query_idx:  查询图片的唯一索引
        topk:       Top-K
        save_path:  保存路径
    """
    image_name = dataset.unique_images[query_idx]
    image_path = dataset._resolve_image_path(image_name)
    img = load_image(image_path)

    gt_indices = dataset.image_to_indices.get(image_name, [])
    gt_set = set(gt_indices)

    # ---- CLIP ----
    clip_scores = clip_sim[query_idx]
    clip_vals, clip_idxs = torch.topk(clip_scores, min(topk, len(clip_scores)))

    # ---- BCSAN ----
    bcsan_scores = bcsan_sim[query_idx]
    bcsan_vals, bcsan_idxs = torch.topk(bcsan_scores, min(topk, len(bcsan_scores)))

    # ---- 绘图 ----
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.35, wspace=0.08)

    # 顶部: 查询图片
    ax_img = fig.add_subplot(gs[0, :])
    ax_img.imshow(img)
    ax_img.axis("off")
    ax_img.set_title(f"Query Image: {image_name}", fontsize=14, fontweight="bold")

    # 下方左: CLIP
    ax_clip = fig.add_subplot(gs[1, 0])
    ax_clip.axis("off")
    ax_clip.set_xlim(0, 1)
    ax_clip.set_ylim(0, 1)
    ax_clip.set_title("CLIP Top-5 Captions", fontsize=13, fontweight="bold", color="#2196F3")

    clip_lines = []
    for rank, (idx, val) in enumerate(zip(clip_idxs, clip_vals)):
        hit = idx.item() in gt_set
        color = "green" if hit else "red"
        cap = dataset.samples[idx.item()]["caption"][:120]
        marker = " [GT]" if hit else ""
        clip_lines.append(f"#{rank+1} [{color.upper()}] score={val:.3f}{marker}\n   {cap}")
    ax_clip.text(
        0.02, 0.95, "\n\n".join(clip_lines),
        fontsize=9, family="monospace",
        verticalalignment="top", transform=ax_clip.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#E3F2FD", alpha=0.9),
    )

    # 下方右: BCSAN
    ax_bcsan = fig.add_subplot(gs[1, 1])
    ax_bcsan.axis("off")
    ax_bcsan.set_xlim(0, 1)
    ax_bcsan.set_ylim(0, 1)
    ax_bcsan.set_title("BCSAN Top-5 Captions", fontsize=13, fontweight="bold", color="#F44336")

    bcsan_lines = []
    for rank, (idx, val) in enumerate(zip(bcsan_idxs, bcsan_vals)):
        hit = idx.item() in gt_set
        color = "green" if hit else "red"
        cap = dataset.samples[idx.item()]["caption"][:120]
        marker = " [GT]" if hit else ""
        bcsan_lines.append(f"#{rank+1} [{color.upper()}] score={val:.3f}{marker}\n   {cap}")
    ax_bcsan.text(
        0.02, 0.95, "\n\n".join(bcsan_lines),
        fontsize=9, family="monospace",
        verticalalignment="top", transform=ax_bcsan.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFEBEE", alpha=0.9),
    )

    plt.suptitle("Image-to-Text: CLIP vs BCSAN", fontsize=16, fontweight="bold", y=1.01)

    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "COMPARE", f"compare_I2T_{query_idx:03d}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [COMPARE I2T] 保存至: {save_path}")


# ============================================================
# CLIP vs BCSAN 并排对比 (Text-to-Image)
# ============================================================


def visualize_compare_t2i(
    dataset: RetrievalDataset,
    clip_sim: torch.Tensor,
    bcsan_sim: torch.Tensor,
    caption_idx: int,
    topk: int = TOPK,
    save_path: str = None,
):
    """
    Text-to-Image 并排对比: CLIP vs BCSAN.

    顶部行: 查询文本
    中间行: CLIP Top-K 图片
    下方行: BCSAN Top-K 图片

    Args:
        dataset:    RetrievalDataset 实例
        clip_sim:   (U, M) CLIP 相似度矩阵
        bcsan_sim:  (U, M) BCSAN 融合相似度矩阵
        caption_idx: 查询 caption 的索引
        topk:       Top-K
        save_path:  保存路径
    """
    caption = dataset.samples[caption_idx]["caption"]
    gt_uid = dataset.get_image_id_for_caption(caption_idx)

    # ---- CLIP ----
    clip_scores = clip_sim[:, caption_idx]
    clip_vals, clip_idxs = torch.topk(clip_scores, min(topk, len(clip_scores)))

    # ---- BCSAN ----
    bcsan_scores = bcsan_sim[:, caption_idx]
    bcsan_vals, bcsan_idxs = torch.topk(bcsan_scores, min(topk, len(bcsan_scores)))

    # ---- 绘图 ----
    fig = plt.figure(figsize=(4 * topk + 2, 8))
    gs = fig.add_gridspec(
        3, topk + 1,
        height_ratios=[0.12, 0.44, 0.44],
        width_ratios=[0.2] + [1] * topk,
        hspace=0.5, wspace=0.03,
    )

    # 顶部: 查询文本
    ax_txt = fig.add_subplot(gs[0, :])
    ax_txt.axis("off")
    ax_txt.text(
        0.5, 0.5, f'Query: "{caption}"',
        fontsize=13, ha="center", va="center",
        transform=ax_txt.transAxes,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="lightyellow", alpha=0.9),
    )

    # 第二行标签: CLIP
    ax_label_clip = fig.add_subplot(gs[1, 0])
    ax_label_clip.axis("off")
    ax_label_clip.text(
        0.5, 0.5, "CLIP",
        fontsize=13, fontweight="bold", color="#2196F3",
        ha="center", va="center", transform=ax_label_clip.transAxes,
    )

    # 第三行标签: BCSAN
    ax_label_bcsan = fig.add_subplot(gs[2, 0])
    ax_label_bcsan.axis("off")
    ax_label_bcsan.text(
        0.5, 0.5, "BCSAN",
        fontsize=13, fontweight="bold", color="#F44336",
        ha="center", va="center", transform=ax_label_bcsan.transAxes,
    )

    # CLIP 图片
    for i, (img_idx, score) in enumerate(zip(clip_idxs, clip_vals)):
        ax = fig.add_subplot(gs[1, i + 1])
        img_name = dataset.unique_images[img_idx.item()]
        img = load_image(dataset._resolve_image_path(img_name))
        ax.imshow(img)
        ax.axis("off")
        hit = img_idx.item() == gt_uid
        border_color = "green" if hit else "red"
        marker = " [GT]" if hit else ""
        ax.set_title(
            f"Rank #{i+1}\nscore={score:.3f}{marker}",
            fontsize=9, color=border_color,
            fontweight="bold" if hit else "normal",
        )
        for spine in ax.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(3 if hit else 1)

    # BCSAN 图片
    for i, (img_idx, score) in enumerate(zip(bcsan_idxs, bcsan_vals)):
        ax = fig.add_subplot(gs[2, i + 1])
        img_name = dataset.unique_images[img_idx.item()]
        img = load_image(dataset._resolve_image_path(img_name))
        ax.imshow(img)
        ax.axis("off")
        hit = img_idx.item() == gt_uid
        border_color = "green" if hit else "red"
        marker = " [GT]" if hit else ""
        ax.set_title(
            f"Rank #{i+1}\nscore={score:.3f}{marker}",
            fontsize=9, color=border_color,
            fontweight="bold" if hit else "normal",
        )
        for spine in ax.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(3 if hit else 1)

    plt.suptitle("Text-to-Image: CLIP vs BCSAN", fontsize=16, fontweight="bold", y=1.01)

    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "COMPARE", f"compare_T2I_{caption_idx:03d}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [COMPARE T2I] 保存至: {save_path}")


# ============================================================
# 论文风格展示 (Paper-style qualitative retrieval figures)
# ============================================================


def visualize_paper_i2t(
    dataset: RetrievalDataset,
    fused_sim: torch.Tensor,
    query_idx: int,
    topk: int = TOPK,
    save_path: str = None,
):
    """
    论文风格 Image-to-Text 检索结果展示（参考 Figure 3 简洁形式）。

    布局:
        左侧上方: Query Image
        左侧下方: 该查询图片对应的 GT 文本
        右侧: Top-K 检索结果图像（3 张一行，5 张分两行）
        不展示 Rank、不展示 ✓/✗ 符号、不展示 score。

    Args:
        dataset:    RetrievalDataset 实例
        fused_sim:  (U, M) BCSAN 最终融合相似度矩阵
        query_idx:  查询图片的唯一索引
        topk:       Top-K
        save_path:  保存路径
    """
    image_name = dataset.unique_images[query_idx]
    query_img = load_image(dataset._resolve_image_path(image_name))

    gt_indices = dataset.image_to_indices.get(image_name, [])
    query_caption = dataset.samples[gt_indices[0]]["caption"] if gt_indices else ""

    topk = min(topk, fused_sim.shape[1])
    _, indices = torch.topk(fused_sim[query_idx], topk)

    fig = plt.figure(figsize=(12, 7.2))
    gs = fig.add_gridspec(
        2, 4,
        width_ratios=[1.15, 1, 1, 1],
        height_ratios=[1, 1],
        wspace=0.02, hspace=0.04,
    )

    # 左侧：查询图片 + 紧贴图片底部的文本
    ax_img = fig.add_subplot(gs[0:2, 0])
    ax_img.imshow(query_img)
    ax_img.axis("off")
    wrapped = textwrap.fill(query_caption, 36)
    ax_img.text(0.5, -0.06, wrapped, fontsize=9, ha="center", va="top",
                transform=ax_img.transAxes, clip_on=False)

    # 右侧：结果图，3 张一行，5 张分两行
    result_positions = [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)]
    for (r, c), idx in zip(result_positions, indices):
        uid = dataset.get_image_id_for_caption(idx.item())
        img_name = dataset.unique_images[uid]
        img = load_image(dataset._resolve_image_path(img_name))
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(img)
        ax.axis("off")

    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "PAPER", f"paper_I2T_{query_idx:03d}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [PAPER I2T] 保存至: {save_path}")


def visualize_paper_t2i(
    dataset: RetrievalDataset,
    fused_sim: torch.Tensor,
    caption_idx: int,
    topk: int = TOPK,
    save_path: str = None,
):
    """
    论文风格 Text-to-Image 检索结果展示（参考 Figure 3 简洁形式）。

    布局:
        左侧上方: Query 对应的 Ground Truth 图像
        左侧下方: Query 文本
        右侧: Top-K 检索图像（3 张一行，5 张分两行）
        不展示 Rank、不展示 ✓/✗ 符号、不展示 score。

    Args:
        dataset:     RetrievalDataset 实例
        fused_sim:   (U, M) BCSAN 最终融合相似度矩阵
        caption_idx: 查询 caption 的索引
        topk:        Top-K
        save_path:   保存路径
    """
    caption = dataset.samples[caption_idx]["caption"]
    gt_uid = dataset.get_image_id_for_caption(caption_idx)
    query_img = load_image(dataset._resolve_image_path(dataset.unique_images[gt_uid]))

    topk = min(topk, fused_sim.shape[0])
    _, indices = torch.topk(fused_sim[:, caption_idx], topk)

    fig = plt.figure(figsize=(12, 7.2))
    gs = fig.add_gridspec(
        2, 4,
        width_ratios=[1.15, 1, 1, 1],
        height_ratios=[1, 1],
        wspace=0.02, hspace=0.04,
    )

    # 左侧：Ground Truth 图像 + 紧贴图片底部的文本
    ax_img = fig.add_subplot(gs[0:2, 0])
    ax_img.imshow(query_img)
    ax_img.axis("off")
    wrapped = textwrap.fill(caption, 36)
    ax_img.text(0.5, -0.06, wrapped, fontsize=9, ha="center", va="top",
                transform=ax_img.transAxes, clip_on=False)

    # 右侧：结果图，3 张一行，5 张分两行
    result_positions = [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)]
    for (r, c), uid in zip(result_positions, indices):
        img_name = dataset.unique_images[uid.item()]
        img = load_image(dataset._resolve_image_path(img_name))
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(img)
        ax.axis("off")

    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "PAPER", f"paper_T2I_{caption_idx:03d}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [PAPER T2I] 保存至: {save_path}")


# ============================================================
# 主程序
# ============================================================


def main():
    print("=" * 70)
    print("  图文检索可视化 (Image-Text Retrieval Visualization)")
    print("=" * 70)

    print(f"\n  配置:")
    print(f"    Model:            {MODEL_NAME}")
    print(f"    Pretrained:       {PRETRAINED_PATH}")
    print(f"    Checkpoint:       {CROSS_MODAL_CHECKPOINT}")
    print(f"    Image Root:       {IMAGE_ROOT}")
    print(f"    Caption File:     {CAPTION_FILE}")
    print(f"    Device:           {DEVICE}")
    print(f"    TopK:             {TOPK}")
    print(f"    Fusion Alpha:     {FUSION_ALPHA}")
    print(f"    Image Queries:    {IMAGE_QUERY_INDICES}")
    print(f"    Text Queries:     {TEXT_QUERY_INDICES}")
    print(f"    Output Dir:       {OUTPUT_DIR}")

    device = torch.device(DEVICE)

    # ========================================================
    # 1. 加载 OpenCLIP
    # ========================================================
    print("\n[1/5] 加载 OpenCLIP 模型 ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED_PATH, device=device,
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    embed_dim = infer_embed_dim(model, tokenizer, device)
    print(f"  embed_dim = {embed_dim}")

    # ========================================================
    # 2. 加载 CrossModalScorer
    # ========================================================
    print("\n[2/5] 加载训练好的 CrossModalScorer ...")
    checkpoint = torch.load(CROSS_MODAL_CHECKPOINT, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    img_token_dim = config.get("img_token_dim", 768)
    txt_token_dim = config.get("txt_token_dim", 512)
    cross_heads = config.get("cross_heads", 8)

    # 从 config 读取消融开关（与训练时保持一致，否则模型结构不匹配）
    use_cross_attention = config.get("use_cross_attention", True)
    use_token_similarity = config.get("use_token_similarity", True)
    use_attention_pool = config.get("use_attention_pool", True)

    if "img_token_dim" not in config:
        with torch.no_grad():
            test_img = torch.randn(
                1, 3, model.visual.image_size[0], model.visual.image_size[1],
                device=device,
            )
            _, test_tokens = model.encode_image(test_img, return_tokens=True)
            img_token_dim = test_tokens.shape[-1]
            test_text_tokens = tokenizer(["test"]).to(device)
            _, test_txt = model.encode_text(test_text_tokens, return_tokens=True)
            txt_token_dim = test_txt.shape[-1]
            print(f"  推断: img_token_dim={img_token_dim}, txt_token_dim={txt_token_dim}")

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    itm_depth = max(
        1,
        len([k for k in state_dict if k.startswith("itm_blocks.") and ".norm.weight" in k]),
    )
    has_refiner = any("img_refiner" in k or "txt_refiner" in k for k in state_dict)

    cross_scorer = CrossModalScorer(
        img_dim=img_token_dim,
        txt_dim=txt_token_dim,
        heads=cross_heads,
        itm_depth=itm_depth,
        enable_per_side_refiner=has_refiner,
        use_cross_attention=use_cross_attention,
        use_token_similarity=use_token_similarity,
        use_attention_pool=use_attention_pool,
    ).to(device)
    print(f"  消融配置: BCA={use_cross_attention}, TLS={use_token_similarity}, AP={use_attention_pool}")
    cross_scorer.load_state_dict(state_dict, strict=False)
    cross_scorer.eval()
    print(f"  CrossModalScorer 加载完成")

    # ========================================================
    # 3. 加载数据集
    # ========================================================
    print("\n[3/5] 加载数据集 ...")
    eval_dataset = RetrievalDataset(
        image_root=IMAGE_ROOT,
        caption_file=CAPTION_FILE,
        transform=preprocess,
    )
    u_eval = len(eval_dataset.unique_images)
    m_eval = len(eval_dataset)
    print(f"  图片数: {u_eval}, Caption数: {m_eval}")

    # ========================================================
    # 4. 提取特征 & 计算相似度矩阵
    # ========================================================
    print("\n[4/5] 提取特征 & 计算相似度矩阵 ...")
    if hasattr(model, 'visual') and hasattr(model.visual, 'output_tokens'):
        model.visual.output_tokens = True

    image_features, image_tokens = extract_unique_image_features(
        model, eval_dataset, BATCH_SIZE, device, embed_dim,
        num_workers=NUM_WORKERS, return_tokens=True,
    )
    text_features, text_tokens = extract_all_text_features(
        model, eval_dataset, tokenizer, BATCH_SIZE, device, embed_dim,
        return_tokens=True,
    )
    print(f"  image_features: {image_features.shape}")
    print(f"  text_features:  {text_features.shape}")

    # ---- CLIP 相似度 ----
    logit_scale = model.logit_scale.exp() if hasattr(model, 'logit_scale') else 1.0
    clip_sim = logit_scale * (image_features @ text_features.T)  # (U, M)
    print(f"  CLIP similarity:     {clip_sim.shape}")

    # ---- Cross-modal 相似度 ----
    image_patches = image_tokens[:, 1:, :]
    del image_tokens
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("  计算 Cross-modal 相似度...")
    with torch.no_grad():
        cross_sim = cross_scorer.forward_retrieval_similarity(
            image_tokens=image_patches,
            text_tokens=text_tokens,
            chunk_size=8,
            show_progress=True,
        )
    print(f"  Cross-modal similarity: {cross_sim.shape}")

    # ---- BCSAN 最终融合相似度 ----
    # BCSAN = alpha * CLIP + (1-alpha) * CrossModalScorer
    fused_sim = FUSION_ALPHA * clip_sim + (1.0 - FUSION_ALPHA) * cross_sim
    print(f"  BCSAN fused similarity: {fused_sim.shape}  (alpha={FUSION_ALPHA})")

    # ========================================================
    # 5. 生成可视化
    # ========================================================
    print(f"\n[5/5] 生成可视化结果 ...\n")

    i2t_clip_dir = os.path.join(OUTPUT_DIR, "I2T", "CLIP")
    i2t_bcsan_dir = os.path.join(OUTPUT_DIR, "I2T", "BCSAN")
    t2i_clip_dir = os.path.join(OUTPUT_DIR, "T2I", "CLIP")
    t2i_bcsan_dir = os.path.join(OUTPUT_DIR, "T2I", "BCSAN")
    paper_dir = os.path.join(OUTPUT_DIR, "PAPER")

    for d in [i2t_clip_dir, i2t_bcsan_dir, t2i_clip_dir, t2i_bcsan_dir, paper_dir]:
        ensure_dir(d)

    # ---- A. I2T: CLIP ----
    print("--- I2T: CLIP ---")
    for q_idx in IMAGE_QUERY_INDICES:
        if q_idx >= u_eval:
            print(f"  [跳过] query_idx={q_idx} >= U={u_eval}")
            continue
        save_path = os.path.join(i2t_clip_dir, f"I2T_{q_idx:03d}.png")
        visualize_i2t(eval_dataset, clip_sim, q_idx, TOPK, save_path, "CLIP")

    # ---- B. I2T: BCSAN (融合后) ----
    print("\n--- I2T: BCSAN (融合) ---")
    for q_idx in IMAGE_QUERY_INDICES:
        if q_idx >= u_eval:
            continue
        save_path = os.path.join(i2t_bcsan_dir, f"I2T_{q_idx:03d}.png")
        visualize_i2t(eval_dataset, fused_sim, q_idx, TOPK, save_path, "BCSAN")

    # ---- C. T2I: CLIP ----
    print("\n--- T2I: CLIP ---")
    for c_idx in TEXT_QUERY_INDICES:
        if c_idx >= m_eval:
            print(f"  [跳过] caption_idx={c_idx} >= M={m_eval}")
            continue
        save_path = os.path.join(t2i_clip_dir, f"T2I_{c_idx:03d}.png")
        visualize_t2i(eval_dataset, clip_sim, c_idx, TOPK, save_path, "CLIP")

    # ---- D. T2I: BCSAN (融合后) ----
    print("\n--- T2I: BCSAN (融合) ---")
    for c_idx in TEXT_QUERY_INDICES:
        if c_idx >= m_eval:
            continue
        save_path = os.path.join(t2i_bcsan_dir, f"T2I_{c_idx:03d}.png")
        visualize_t2i(eval_dataset, fused_sim, c_idx, TOPK, save_path, "BCSAN")

    # ---- E. 论文风格 I2T（仅结果图片） ----
    print("\n--- PAPER: I2T (Result Images) ---")
    for q_idx in IMAGE_QUERY_INDICES:
        if q_idx >= u_eval:
            continue
        save_path = os.path.join(paper_dir, f"paper_I2T_{q_idx:03d}.png")
        visualize_paper_i2t(eval_dataset, fused_sim, q_idx, TOPK, save_path)

    # ---- F. 论文风格 T2I（仅结果图片） ----
    print("\n--- PAPER: T2I (Result Images) ---")
    for c_idx in TEXT_QUERY_INDICES:
        if c_idx >= m_eval:
            continue
        save_path = os.path.join(paper_dir, f"paper_T2I_{c_idx:03d}.png")
        visualize_paper_t2i(eval_dataset, fused_sim, c_idx, TOPK, save_path)

    # ========================================================
    # 完成
    # ========================================================
    print("\n" + "=" * 70)
    print(f"  可视化完成! 结果保存在: {os.path.abspath(OUTPUT_DIR)}")
    print(f"  目录结构:")
    print(f"    {OUTPUT_DIR}/")
    print(f"    ├── I2T/CLIP/       # CLIP Image-to-Text")
    print(f"    ├── I2T/BCSAN/      # BCSAN Image-to-Text (融合后)")
    print(f"    ├── T2I/CLIP/       # CLIP Text-to-Image")
    print(f"    ├── T2I/BCSAN/      # BCSAN Text-to-Image (融合后)")
    print(f"    └── PAPER/          # 论文风格展示: BCSAN 检索结果图片")
    print("=" * 70)


if __name__ == "__main__":
    main()