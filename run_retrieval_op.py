#!/usr/bin/env python
"""
加载训练好的 CrossModalScorer checkpoint 并运行 OPMMD 图文检索评估。

用法:
    cd GAI-open_clip
    python run_retrieval_op.py
"""

import sys
import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

# 将 src/ 加入 Python path
_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import open_clip
from open_clip.cross_modal import CrossModalScorer, compute_cross_modal_similarity

# 直接导入 retrieval_op 中的工具函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieval_op import (
    RetrievalDataset,
    extract_unique_image_features,
    extract_all_text_features,
    compute_recall_from_similarity,
    infer_embed_dim,
)

# ============================================================
# 配置
# ============================================================
MODEL_NAME = "ViT-B-32"
PRETRAINED_PATH = "/root/autodl-tmp/ViT-B-32.pt"
CROSS_MODAL_CHECKPOINT = "/root/autodl-tmp/GAI-open_clip/checkpoints/cross_modal_opmmd_noAP.pth"
IMAGE_ROOT = "/root/autodl-tmp/OPMMD/image/test"
CAPTION_FILE = "/root/autodl-tmp/dataset/test.json"
BATCH_SIZE = 64
NUM_WORKERS = 0
DEVICE = "cuda"
KS = [1, 5, 10]

# Alpha 消融实验: 0 = Cross-only, 1 = CLIP-only
ALPHAS = [0, 0.25, 0.5, 0.75, 1.0]

OUTPUT = "/root/autodl-tmp/checkpoints/retrieval_results_opmmd_noAP.json"


def main():
    device = torch.device(DEVICE)
    print(f"使用设备: {device}")

    # ========================================================
    # 1. 加载 CLIP 模型
    # ========================================================
    print("\n[1/6] 加载 OpenCLIP 模型 ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED_PATH,
        device=device,
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    embed_dim = infer_embed_dim(model, tokenizer, device)
    print(f"  模型加载完成, embed_dim={embed_dim}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ========================================================
    # 2. 加载训练好的 CrossModalScorer
    # ========================================================
    print("\n[2/6] 加载训练好的 CrossModalScorer ...")
    checkpoint = torch.load(CROSS_MODAL_CHECKPOINT, map_location=device, weights_only=False)

    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Checkpoint loss: {checkpoint.get('loss', 'N/A'):.4f}")
    print(f"  Checkpoint loss_type: {checkpoint.get('loss_type', 'N/A')}")
    config = checkpoint.get("config", {})
    print(f"  训练配置: {json.dumps({k: v for k, v in config.items() if k != 'pretrained'}, indent=2, ensure_ascii=False)}")

    # 从 config 或实际数据推断维度
    img_token_dim = config.get("img_token_dim", 768)
    txt_token_dim = config.get("txt_token_dim", 512)
    cross_heads = config.get("cross_heads", 8)

    # 如果 config 没有维度信息，通过实际推理获取
    if "img_token_dim" not in config:
        with torch.no_grad():
            test_img = torch.randn(1, 3, model.visual.image_size[0], model.visual.image_size[1], device=device)
            _, test_tokens = model.encode_image(test_img, return_tokens=True)
            img_token_dim = test_tokens.shape[-1]
            test_text_tokens = tokenizer(["test"]).to(device)
            _, test_txt = model.encode_text(test_text_tokens, return_tokens=True)
            txt_token_dim = test_txt.shape[-1]
            print(f"  推断维度: img_token_dim={img_token_dim}, txt_token_dim={txt_token_dim}")

    # 从 state_dict 推断 itm_depth 和 enable_per_side_refiner
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    itm_depth = max(1, len([k for k in state_dict if k.startswith("itm_blocks.") and ".norm.weight" in k]))
    has_refiner = any("img_refiner" in k or "txt_refiner" in k for k in state_dict)

    cross_scorer = CrossModalScorer(
        img_dim=img_token_dim,
        txt_dim=txt_token_dim,
        heads=cross_heads,
        itm_depth=itm_depth,
        enable_per_side_refiner=has_refiner,
    ).to(device)

    missing_keys, unexpected_keys = cross_scorer.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"  [信息] state_dict 缺失 keys ({len(missing_keys)} 个), 使用随机初始化")
    if unexpected_keys:
        non_itm_unexpected = [k for k in unexpected_keys if not k.startswith("itm_proj.") and not k.startswith("itm_blocks.") and not k.startswith("itm_head.")]
        if non_itm_unexpected:
            print(f"  [警告] state_dict 多余 keys ({len(non_itm_unexpected)} 个): {non_itm_unexpected[:5]}")

    cross_scorer.eval()
    print(f"  CrossModalScorer 加载完成, 参数量: {sum(p.numel() for p in cross_scorer.parameters()):,}")

    # ========================================================
    # 3. 加载数据集
    # ========================================================
    print("\n[3/6] 加载 OPMMD 测试集 ...")
    eval_dataset = RetrievalDataset(
        image_root=IMAGE_ROOT,
        caption_file=CAPTION_FILE,
        transform=preprocess,
    )
    u_eval = len(eval_dataset.unique_images)
    m_eval = len(eval_dataset)
    print(f"  唯一图片: {u_eval}")
    print(f"  captions: {m_eval}")
    print(f"  平均每图 caption: {m_eval / u_eval:.1f}")

    # ========================================================
    # 4. 提取特征
    # ========================================================
    print("\n[4/6] 提取特征 (含 token embeddings) ...")

    if hasattr(model, 'visual') and hasattr(model.visual, 'output_tokens'):
        model.visual.output_tokens = True

    image_features, image_tokens = extract_unique_image_features(
        model, eval_dataset, BATCH_SIZE, device, embed_dim,
        num_workers=NUM_WORKERS,
        return_tokens=True,
    )
    text_features, text_tokens = extract_all_text_features(
        model, eval_dataset, tokenizer, BATCH_SIZE, device, embed_dim,
        return_tokens=True,
    )
    print(f"  image_features: {image_features.shape}  (U={u_eval}, D={embed_dim})")
    print(f"  image_tokens:   {image_tokens.shape}")
    print(f"  text_features:  {text_features.shape}   (M={m_eval}, D={embed_dim})")
    print(f"  text_tokens:    {text_tokens.shape}")

    # ========================================================
    # 5. 计算 Recall@K（含 alpha 消融实验）
    # ========================================================
    print("\n[5/6] 计算 Recall@K (含融合消融实验) ...")

    logit_scale = model.logit_scale.exp() if hasattr(model, 'logit_scale') else 1.0

    # --- A. 纯 CLIP (baseline) ---
    clip_similarity = logit_scale * (image_features @ text_features.T)
    clip_metrics = compute_recall_from_similarity(clip_similarity, eval_dataset, ks=KS)
    print("\n  === 纯 CLIP (baseline) ===")
    for k in KS:
        print(f"  Image-to-Text R@{k}:  {clip_metrics[f'I2T_R@{k}']:.2f}%")
        print(f"  Text-to-Image R@{k}:  {clip_metrics[f'T2I_R@{k}']:.2f}%")
    print(f"  Mean Recall:          {clip_metrics['Mean_Recall']:.2f}%")

    # --- B. Cross-modal similarity ---
    print("\n  === 计算 Cross-modal 分数 ===")
    image_patches = image_tokens[:, 1:, :]
    del image_tokens
    if device.type == "cuda":
        torch.cuda.empty_cache()

    with torch.no_grad():
        cross_sim = cross_scorer.forward_retrieval_similarity(
            image_tokens=image_patches,
            text_tokens=text_tokens,
            chunk_size=8,
            show_progress=True,
        )

    # --- C. Cross-only ---
    cross_only_metrics = compute_recall_from_similarity(cross_sim, eval_dataset, ks=KS)
    print("\n  === Cross-only (纯跨模态，无 CLIP 融合) ===")
    for k in KS:
        print(f"  Image-to-Text R@{k}:  {cross_only_metrics[f'I2T_R@{k}']:.2f}%")
        print(f"  Text-to-Image R@{k}:  {cross_only_metrics[f'T2I_R@{k}']:.2f}%")
    print(f"  Mean Recall:          {cross_only_metrics['Mean_Recall']:.2f}%")

    # --- D. Alpha 消融 ---
    print("\n  === Alpha 消融实验 (融合比例) ===")
    print(f"  {'Alpha':<8} {'含义':<20} {'Mean Recall':>12}")
    print(f"  {'-'*8} {'-'*20} {'-'*12}")

    alpha_results = {}
    for alpha in ALPHAS:
        if alpha == 0:
            metrics = cross_only_metrics
            label = "Cross-only"
        elif alpha == 1.0:
            metrics = clip_metrics
            label = "CLIP-only"
        else:
            fused_similarity = alpha * clip_similarity + (1 - alpha) * cross_sim
            metrics = compute_recall_from_similarity(fused_similarity, eval_dataset, ks=KS)
            label = f"CLIP+Cross (alpha={alpha})"

        alpha_results[f"alpha_{alpha}"] = {
            "label": label,
            "metrics": {k: round(v, 4) for k, v in metrics.items()},
        }
        print(f"  {alpha:<8} {label:<20} {metrics['Mean_Recall']:>11.2f}%")

    # 自动选取最优 alpha
    best_alpha = None
    best_mean_recall = -1
    best_fusion_metrics = None
    for alpha in ALPHAS:
        if alpha == 0 or alpha == 1.0:
            continue
        mr = alpha_results[f"alpha_{alpha}"]["metrics"]["Mean_Recall"]
        if mr > best_mean_recall:
            best_mean_recall = mr
            best_alpha = alpha
            best_fusion_metrics = alpha_results[f"alpha_{alpha}"]["metrics"]

    # ========================================================
    # 6. 输出总结
    # ========================================================
    print("\n" + "=" * 80)
    if best_alpha is not None:
        print(f"  最优融合 alpha = {best_alpha}, Mean Recall = {best_mean_recall:.2f}%")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("  最终结果对比")
    print("=" * 80)
    if best_alpha is not None:
        print(f"  {'Metric':<20} {'CLIP':>10} {'Cross-only':>12} {'Fusion (a=' + str(best_alpha) + ')':>16}")
        print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*16}")
        for k in KS:
            print(f"  I2T R@{k:<16} {clip_metrics[f'I2T_R@{k}']:>9.2f}% {cross_only_metrics[f'I2T_R@{k}']:>11.2f}% {best_fusion_metrics[f'I2T_R@{k}']:>15.2f}%")
            print(f"  T2I R@{k:<16} {clip_metrics[f'T2I_R@{k}']:>9.2f}% {cross_only_metrics[f'T2I_R@{k}']:>11.2f}% {best_fusion_metrics[f'T2I_R@{k}']:>15.2f}%")
        print(f"  Mean Recall{'':<9} {clip_metrics['Mean_Recall']:>9.2f}% {cross_only_metrics['Mean_Recall']:>11.2f}% {best_fusion_metrics['Mean_Recall']:>15.2f}%")
    else:
        print(f"  {'Metric':<20} {'CLIP':>10} {'Cross-only':>12}")
        print(f"  {'-'*20} {'-'*10} {'-'*12}")
        for k in KS:
            print(f"  I2T R@{k:<16} {clip_metrics[f'I2T_R@{k}']:>9.2f}% {cross_only_metrics[f'I2T_R@{k}']:>11.2f}%")
            print(f"  T2I R@{k:<16} {clip_metrics[f'T2I_R@{k}']:>9.2f}% {cross_only_metrics[f'T2I_R@{k}']:>11.2f}%")
        print(f"  Mean Recall{'':<9} {clip_metrics['Mean_Recall']:>9.2f}% {cross_only_metrics['Mean_Recall']:>11.2f}%")
    print("=" * 80)

    # 保存结果
    output_data = {
        "config": {
            "model": MODEL_NAME,
            "pretrained": PRETRAINED_PATH,
            "cross_modal_checkpoint": CROSS_MODAL_CHECKPOINT,
            "cross_heads": cross_heads,
            "dataset": "OPMMD_test",
            "image_root": IMAGE_ROOT,
            "caption_file": CAPTION_FILE,
            "num_images": u_eval,
            "num_captions": m_eval,
            "embed_dim": embed_dim,
        },
        "clip_results": {k: round(v, 4) for k, v in clip_metrics.items()},
        "cross_only_results": {k: round(v, 4) for k, v in cross_only_metrics.items()},
    }
    if best_alpha is not None:
        output_data["best_fusion_alpha"] = best_alpha
        output_data["fusion_results"] = {k: round(v, 4) for k, v in best_fusion_metrics.items()}
    output_data["alpha_sweep"] = alpha_results

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存至: {OUTPUT}")


if __name__ == "__main__":
    main()