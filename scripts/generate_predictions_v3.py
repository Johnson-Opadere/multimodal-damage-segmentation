#!/usr/bin/env python3
"""
Generate predicted masks (.npy) for test split
Project 1 — GATED DIFF v3
"""

import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# 🔥 IMPORTANT: ensures model loads correctly
import src.models.dual_branch_diff_gated_segmentation_model_v3

# ========================
# PATHS
# ========================
MODEL_PATH = "/mnt/ebs-data/cv_project1_new/experiments_v2/dual_branch_diff_gated_v3_20260406-224335/best_model.keras"

DATA_ROOT = "/mnt/ebs-data/cv_project1_new/data/normalized_data/test"

RGB_PRE_DIR  = os.path.join(DATA_ROOT, "rgb_pre_norm")
RGB_POST_DIR = os.path.join(DATA_ROOT, "rgb_post_norm")
SAR_PRE_DIR  = os.path.join(DATA_ROOT, "sar_pre_norm")
SAR_POST_DIR = os.path.join(DATA_ROOT, "sar_post_norm")

OUTPUT_DIR = "/mnt/ebs-data/cv_project1_new/predictions_v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================
# HELPERS
# ========================
def extract_id(filename):
    return filename.replace("_post_disaster_norm.npy", "")

def load_rgb(path):
    return np.load(path).astype(np.float32)  # (512,512,3)

def load_sar(path):
    return np.load(path).astype(np.float32)  # (512,512,8)

# ========================
# GET IDS
# ========================
rgb_post_files = os.listdir(RGB_POST_DIR)

ids = [extract_id(f) for f in rgb_post_files]

print(f"\nFound {len(ids)} samples")

# ========================
# LOAD MODEL
# ========================
print("\n[INFO] Loading model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    safe_mode=False
)
print("[INFO] Model loaded.\n")

# ========================
# INFERENCE LOOP
# ========================
for base in tqdm(ids):

    try:
        rgb_pre_path  = os.path.join(RGB_PRE_DIR,  f"{base}_pre_disaster_norm.npy")
        rgb_post_path = os.path.join(RGB_POST_DIR, f"{base}_post_disaster_norm.npy")

        sar_pre_path  = os.path.join(SAR_PRE_DIR,  f"{base}_pre_sar_norm.npy")
        sar_post_path = os.path.join(SAR_POST_DIR, f"{base}_post_sar_norm.npy")

        if not all(map(os.path.exists, [
            rgb_pre_path, rgb_post_path,
            sar_pre_path, sar_post_path
        ])):
            continue

        # ========================
        # LOAD DATA
        # ========================
        rgb_pre  = load_rgb(rgb_pre_path)
        rgb_post = load_rgb(rgb_post_path)

        sar_pre  = load_sar(sar_pre_path)
        sar_post = load_sar(sar_post_path)

        # ========================
        # BUILD INPUT (22 CHANNELS)
        # ========================
        x = np.concatenate([
            rgb_pre,
            rgb_post,
            sar_pre,
            sar_post
        ], axis=-1)

        x = np.expand_dims(x, axis=0)  # (1,512,512,22)

        # ========================
        # PREDICT
        # ========================
        logits = model(x, training=False)

        pred = tf.argmax(logits, axis=-1)[0].numpy().astype(np.uint8)

        # ========================
        # SAVE
        # ========================
        out_path = os.path.join(
            OUTPUT_DIR,
            f"{base}_pred_mask.npy"
        )

        np.save(out_path, pred)

    except Exception as e:
        print(f"❌ Error on {base}: {e}")

print("\n✅ Done: All predictions saved\n")
