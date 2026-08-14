#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ========================
# PATH
# ========================
PRED_DIR = "/mnt/ebs-data/cv_project1_new/predictions_v3"
NUM_VIS = 6

# ========================
# LOAD FILES
# ========================
files = [f for f in os.listdir(PRED_DIR) if f.endswith("_pred_mask.npy")]
files.sort()

print(f"\nFound {len(files)} prediction files\n")

# ========================
# METRICS
# ========================
empty = 0
low_damage = 0
high_damage = 0
shape_issues = 0

class_counter = Counter()
damage_ratios = []

# ========================
# LOOP
# ========================
for f in files:
    path = os.path.join(PRED_DIR, f)
    pred = np.load(path)

    # ---------- SHAPE ----------
    if pred.shape != (512, 512):
        print(f"⚠️ Shape issue: {f} → {pred.shape}")
        shape_issues += 1
        continue

    # ---------- VALUE RANGE ----------
    if pred.max() > 3 or pred.min() < 0:
        print(f"⚠️ Invalid class values in {f}")

    # ---------- EMPTY ----------
    if np.all(pred == 0):
        empty += 1

    # ---------- DAMAGE RATIO ----------
    dmg_ratio = np.mean(pred > 0)
    damage_ratios.append(dmg_ratio)

    if dmg_ratio < 0.01:
        low_damage += 1
    if dmg_ratio > 0.3:
        high_damage += 1

    # ---------- CLASS DISTRIBUTION ----------
    u, c = np.unique(pred, return_counts=True)
    for ui, ci in zip(u, c):
        class_counter[ui] += ci

# ========================
# SUMMARY
# ========================
print("\n===== VALIDATION SUMMARY =====")

print(f"Total files: {len(files)}")
print(f"Empty predictions: {empty}")
print(f"Low-damage (<1%): {low_damage}")
print(f"High-damage (>30%): {high_damage}")
print(f"Shape issues: {shape_issues}")

print("\nDamage ratio stats:")
print(f"Mean: {np.mean(damage_ratios):.4f}")
print(f"Median: {np.median(damage_ratios):.4f}")
print(f"Max: {np.max(damage_ratios):.4f}")

print("\nClass distribution (pixel-wise):")
total_pixels = sum(class_counter.values())

for k in sorted(class_counter.keys()):
    pct = 100 * class_counter[k] / total_pixels
    print(f"Class {k}: {class_counter[k]:,} ({pct:.2f}%)")

# ========================
# HISTOGRAM
# ========================
plt.figure()
plt.hist(damage_ratios, bins=30)
plt.title("Damage Ratio Distribution")
plt.xlabel("Damage Ratio")
plt.ylabel("Count")
plt.savefig("damage_ratio_hist.png", dpi=200)
plt.close()

print("\nSaved: damage_ratio_hist.png")

# ========================
# VISUAL SAMPLES
# ========================
print("\nShowing representative samples...")

# sort by damage ratio
sorted_files = sorted(zip(damage_ratios, files), reverse=True)

samples = (
    sorted_files[:2] +                          # BEST
    sorted_files[len(files)//2:len(files)//2+2] +  # TYPICAL
    sorted_files[-2:]                           # FAILURE
)

for i, (_, f) in enumerate(samples):
    pred = np.load(os.path.join(PRED_DIR, f))

    plt.figure(figsize=(4,4))
    plt.imshow(pred, cmap="jet")
    plt.title(f)
    plt.axis("off")

plt.show()
