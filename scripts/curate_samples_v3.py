#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

# ========================
# PATHS
# ========================
PRED_DIR = "/mnt/ebs-data/cv_project1_new/predictions_v3"
RGB_DIR  = "/mnt/ebs-data/cv_project1_new/data/normalized_data/test/rgb_post_norm"

OUT_DIR = "/mnt/ebs-data/cv_project1_new/curated_v3"

BEST_DIR    = os.path.join(OUT_DIR, "best")
TYPICAL_DIR = os.path.join(OUT_DIR, "typical")
FAIL_DIR    = os.path.join(OUT_DIR, "failure")

for d in [BEST_DIR, TYPICAL_DIR, FAIL_DIR]:
    os.makedirs(d, exist_ok=True)

TOP_K = 10

# ========================
# HELPERS
# ========================

def load_rgb(path):
    rgb = np.load(path)
    return np.clip(rgb, 0, 1)

def overlay_pred(rgb, pred):
    vis = rgb.copy()
    vis *= 0.6

    vis[pred == 1] = [1, 1, 0]
    vis[pred == 2] = [1, 0.5, 0]
    vis[pred == 3] = [1, 0, 0]

    return vis

def largest_component_size(mask):
    binary = (mask > 0).astype(np.int32)

    if binary.sum() == 0:
        return 0, 0

    labeled, num = label(binary)

    if num == 0:
        return 0, 0

    sizes = []
    for i in range(1, num + 1):
        sizes.append(np.sum(labeled == i))

    return max(sizes), num  # (largest_component, num_components)

# ========================
# LOAD FILES
# ========================
pred_files = [f for f in os.listdir(PRED_DIR) if f.endswith("_pred_mask.npy")]
pred_files.sort()

print(f"Found {len(pred_files)} prediction files")

records = []

# ========================
# SCORE
# ========================
for f in pred_files:

    base = f.replace("_pred_mask.npy", "")
    pred_path = os.path.join(PRED_DIR, f)

    rgb_name = base + "_post_disaster_norm.npy"
    rgb_path = os.path.join(RGB_DIR, rgb_name)

    if not os.path.exists(rgb_path):
        continue

    pred = np.load(pred_path)

    dmg_ratio = np.mean(pred > 0)

    lcc, num_components = largest_component_size(pred)

    records.append({
        "base": base,
        "pred_path": pred_path,
        "rgb_path": rgb_path,
        "dmg_ratio": dmg_ratio,
        "lcc": lcc,
        "num_components": num_components
    })

print(f"Usable samples: {len(records)}")

# ========================
# SORTING
# ========================

best = sorted(
    records,
    key=lambda x: (x["dmg_ratio"], x["lcc"]),
    reverse=True
)

records_sorted = sorted(records, key=lambda x: x["dmg_ratio"])
mid = len(records_sorted) // 2
typical = records_sorted[max(0, mid-20):mid+20]

failure = sorted(
    records,
    key=lambda x: (x["dmg_ratio"], -x["lcc"])
)

# ========================
# SAVE
# ========================

def save_samples(samples, out_dir, label):

    count = 0

    for s in samples:

        if count >= TOP_K:
            break

        pred = np.load(s["pred_path"])
        rgb  = load_rgb(s["rgb_path"])
        overlay = overlay_pred(rgb, pred)

        plt.figure(figsize=(9,3))

        plt.subplot(1,3,1)
        plt.imshow(rgb)
        plt.title("RGB")
        plt.axis("off")

        plt.subplot(1,3,2)
        plt.imshow(pred, cmap="jet")
        plt.title("Prediction")
        plt.axis("off")

        plt.subplot(1,3,3)
        plt.imshow(overlay)
        plt.title("Overlay")
        plt.axis("off")

        out_path = os.path.join(out_dir, f"{label}_{count}_{s['base']}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

        count += 1

    print(f"Saved {count} → {out_dir}")

# ========================
# RUN
# ========================
print("\nSaving BEST...")
save_samples(best, BEST_DIR, "best")

print("\nSaving TYPICAL...")
save_samples(typical, TYPICAL_DIR, "typical")

print("\nSaving FAILURE...")
save_samples(failure, FAIL_DIR, "failure")

print("\nDone.")
