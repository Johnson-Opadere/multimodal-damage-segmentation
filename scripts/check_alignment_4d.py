#!/usr/bin/env python3
"""
check_alignment_4d.py

4D Multimodal Alignment Sanity Check

For each sampled event:
  • PRE: RGB, SAR
  • POST: RGB, SAR, MASK
  • DIFF: SAR(pre vs post)
  • OVERLAYS:
        RGB + SAR
        RGB + MASK
        RGB + SAR + MASK
        RGB + DIFF

Usage:
  python check_alignment_4d.py --data-root /path/to/data

Example:
  python check_alignment_4d.py --data-root /mnt/ebs-data/cv_project1_new/data

If --data-root is not provided, defaults to ./data
"""

import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt


# ========================
# CLI
# ========================
def parse_args():
    parser = argparse.ArgumentParser(description="4D Alignment Sanity Check")
    parser.add_argument(
        "--data-root",
        type=str,
        default=os.getenv("DATA_ROOT", "./data"),
        help="Root directory containing normalized_data/ and mask/",
    )
    parser.add_argument("--n-events", type=int, default=2)
    return parser.parse_args()


# ========================
# Helpers
# ========================
def load_rgb(path):
    rgb = np.load(path)
    vis = np.zeros_like(rgb)

    for c in range(3):
        vmin, vmax = np.percentile(rgb[..., c], [2, 98])
        vis[..., c] = np.clip((rgb[..., c] - vmin) / (vmax - vmin + 1e-6), 0, 1)

    return vis


def sar_to_vis(sar):
    band = sar[..., 0]
    vmin, vmax = np.percentile(band, [5, 95])
    return np.clip((band - vmin) / (vmax - vmin + 1e-6), 0, 1)


def compute_sar_diff(pre_sar, post_sar):
    diff = np.abs(pre_sar[..., 0] - post_sar[..., 0])
    vmin, vmax = np.percentile(diff, [5, 95])
    return np.clip((diff - vmin) / (vmax - vmin + 1e-6), 0, 1)


def colorize_mask(mask):
    cmap = np.zeros((*mask.shape, 3), dtype=np.float32)
    cmap[mask == 1] = [1, 1, 0]
    cmap[mask == 2] = [1, 0.5, 0]
    cmap[mask == 3] = [1, 0, 0]
    return cmap


def extract_event_id_rgb(fname):
    return fname.replace("_pre_disaster_norm.npy", "")


def extract_event_id_sar(fname):
    return fname.replace("_pre_sar_norm.npy", "")


def extract_event_id_mask(fname):
    return fname.replace("_post_disaster_mask.npy", "")


# ========================
# Main
# ========================
def main():
    args = parse_args()

    DATA_ROOT = args.data_root

    RGB_PRE_DIR = os.path.join(DATA_ROOT, "normalized_data/train/rgb_pre_norm")
    SAR_PRE_DIR = os.path.join(DATA_ROOT, "normalized_data/train/sar_pre_norm")
    RGB_POST_DIR = os.path.join(DATA_ROOT, "normalized_data/train/rgb_post_norm")
    SAR_POST_DIR = os.path.join(DATA_ROOT, "normalized_data/train/sar_post_norm")
    MASK_DIR = os.path.join(DATA_ROOT, "mask/train")

    ALPHA_SAR = 0.4
    ALPHA_MASK = 0.35
    ALPHA_DIFF = 0.5

    # -----------------------
    # Build event intersection
    # -----------------------
    rgb_ids = {
        extract_event_id_rgb(f)
        for f in os.listdir(RGB_PRE_DIR)
        if f.endswith("_pre_disaster_norm.npy")
    }

    sar_ids = {
        extract_event_id_sar(f)
        for f in os.listdir(SAR_PRE_DIR)
        if f.endswith("_pre_sar_norm.npy")
    }

    mask_ids = {
        extract_event_id_mask(f)
        for f in os.listdir(MASK_DIR)
        if f.endswith("_mask.npy")
    }

    event_ids = sorted(rgb_ids & sar_ids & mask_ids)

    print("\n[DEBUG]")
    print(f"RGB events   : {len(rgb_ids)}")
    print(f"SAR events   : {len(sar_ids)}")
    print(f"Mask events  : {len(mask_ids)}")
    print(f"Common events: {len(event_ids)}")

    if not event_ids:
        raise RuntimeError("No matching events across modalities.")

    selected_events = random.sample(event_ids, min(args.n_events, len(event_ids)))

    print("\n[4D Alignment Check]\n")

    for event_id in selected_events:

        print(f"\n=== EVENT: {event_id} ===")

        paths = {
            "pre_rgb": os.path.join(RGB_PRE_DIR, f"{event_id}_pre_disaster_norm.npy"),
            "pre_sar": os.path.join(SAR_PRE_DIR, f"{event_id}_pre_sar_norm.npy"),
            "post_rgb": os.path.join(RGB_POST_DIR, f"{event_id}_post_disaster_norm.npy"),
            "post_sar": os.path.join(SAR_POST_DIR, f"{event_id}_post_sar_norm.npy"),
            "mask": os.path.join(MASK_DIR, f"{event_id}_post_disaster_mask.npy"),
        }

        pre_rgb = load_rgb(paths["pre_rgb"])
        post_rgb = load_rgb(paths["post_rgb"])
        pre_sar = np.load(paths["pre_sar"])
        post_sar = np.load(paths["post_sar"])
        mask = np.load(paths["mask"])

        pre_sar_vis = sar_to_vis(pre_sar)
        post_sar_vis = sar_to_vis(post_sar)
        sar_diff = compute_sar_diff(pre_sar, post_sar)
        mask_color = colorize_mask(mask)

        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle(f"{event_id} — 4D Check", fontsize=12)

        axes[0, 0].imshow(pre_rgb); axes[0, 0].set_title("Pre RGB")
        axes[0, 1].imshow(pre_sar_vis, cmap="gray"); axes[0, 1].set_title("Pre SAR")
        axes[0, 2].imshow(post_rgb); axes[0, 2].set_title("Post RGB")
        axes[0, 3].imshow(post_sar_vis, cmap="gray"); axes[0, 3].set_title("Post SAR")

        axes[1, 0].imshow(mask, cmap="gray"); axes[1, 0].set_title("Mask")
        axes[1, 1].imshow(mask_color); axes[1, 1].set_title("Mask (color)")
        axes[1, 2].imshow(sar_diff, cmap="hot"); axes[1, 2].set_title("SAR Diff")
        axes[1, 3].imshow(post_rgb); axes[1, 3].imshow(sar_diff, cmap="hot", alpha=ALPHA_DIFF)
        axes[1, 3].set_title("RGB + SAR Diff")

        axes[2, 0].imshow(post_rgb); axes[2, 0].imshow(post_sar_vis, cmap="hot", alpha=ALPHA_SAR)
        axes[2, 0].set_title("RGB + SAR")

        axes[2, 1].imshow(post_rgb); axes[2, 1].imshow(mask_color, alpha=ALPHA_MASK)
        axes[2, 1].set_title("RGB + Mask")

        axes[2, 2].imshow(post_rgb)
        axes[2, 2].imshow(post_sar_vis, cmap="hot", alpha=ALPHA_SAR)
        axes[2, 2].imshow(mask_color, alpha=ALPHA_MASK)
        axes[2, 2].set_title("RGB + SAR + Mask")

        axes[2, 3].imshow(pre_rgb)
        axes[2, 3].imshow(pre_sar_vis, cmap="hot", alpha=ALPHA_SAR)
        axes[2, 3].set_title("Pre RGB + SAR")

        for ax in axes.flatten():
            ax.axis("off")

        plt.tight_layout()
        plt.show()

        diff_val = np.mean(np.abs(pre_sar - post_sar))
        print(f"[DIFF] Mean SAR diff: {diff_val:.6f}")

        print(f"Checked event: {event_id}")

    print("\n4D sanity check complete.")
    print("Look for: alignment + meaningful SAR diff + mask correctness.")


if __name__ == "__main__":
    main()