#!/usr/bin/env python3
"""
================================================================================
Project 1 — FINAL Test Evaluation (GATED DIFF v3, NO TTA, TRAINING-METRIC ALIGNED)
================================================================================

Purpose
-------
Evaluate Efficient Gated Diff v3 model using EXACT SAME metrics as training.

Ensures:
    ✔ apples-to-apples comparison with validation
    ✔ correct mIoU reporting
    ✔ direct comparison with v2

--------------------------------------------------------------------------------
Run
--------------------------------------------------------------------------------

PYTHONPATH=. python scripts_v2/evaluate_test_v3.py \
    --model experiments_v2/dual_branch_diff_gated_v3_<timestamp>/best_model.keras

================================================================================
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# 🔥 IMPORTANT: import v3 model module (ensures safe loading)
import src.models.dual_branch_diff_gated_segmentation_model_v3

# Dataset (SAME AS v2)
from src.data.tfrecord_loader import get_dataset

# Metrics (IDENTICAL to training)
from src.metrics.iou import MeanIoUWrapper
from src.metrics.per_class_iou import PerClassIoU


# =========================================================
# GPU Setup
# =========================================================
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# =========================================================
# Damage-only IoU (same as v2)
# =========================================================
def compute_damage_iou(y_true, y_pred):
    true_damage = (y_true > 0).astype(np.uint8)
    pred_damage = (y_pred > 0).astype(np.uint8)

    intersection = np.logical_and(true_damage, pred_damage).sum()
    union = np.logical_or(true_damage, pred_damage).sum()

    if union == 0:
        return None

    return intersection / (union + 1e-7)


# =========================================================
# Main
# =========================================================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError("Model not found")

    experiment_dir = os.path.dirname(args.model)

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------
    print("\n[INFO] Loading v3 model...")

    model = tf.keras.models.load_model(
        args.model,
        compile=False,
        safe_mode=False   # safe even if no Lambda (consistent with v2)
    )

    print("[INFO] Model loaded.\n")

    # -----------------------------------------------------
    # Dataset (IDENTICAL to v2)
    # -----------------------------------------------------
    print("[INFO] Building test dataset...")

    test_ds = get_dataset(
        "tfrecords_v2/test/raw",
        batch_size=4,   # memory-safe
        shuffle=False
    )

    print("[INFO] Dataset ready.\n")

    # -----------------------------------------------------
    # Metrics (IDENTICAL to training)
    # -----------------------------------------------------
    mean_iou_metric = MeanIoUWrapper(num_classes=4)

    per_class_0 = PerClassIoU(num_classes=4, class_id=0)
    per_class_1 = PerClassIoU(num_classes=4, class_id=1)
    per_class_2 = PerClassIoU(num_classes=4, class_id=2)
    per_class_3 = PerClassIoU(num_classes=4, class_id=3)

    # -----------------------------------------------------
    # Damage stats
    # -----------------------------------------------------
    total_patches = 0
    gt_damage_patches = 0
    damage_only_ious = []

    print("[INFO] Running inference...\n")

    # -----------------------------------------------------
    # Inference loop
    # -----------------------------------------------------
    for images, masks in tqdm(test_ds):

        logits = model(images, training=False)

        # 🔥 CRITICAL: metrics use logits (NOT argmax)
        mean_iou_metric.update_state(masks, logits)
        per_class_0.update_state(masks, logits)
        per_class_1.update_state(masks, logits)
        per_class_2.update_state(masks, logits)
        per_class_3.update_state(masks, logits)

        preds = tf.argmax(logits, axis=-1)

        masks_np = masks.numpy()
        preds_np = preds.numpy()

        for i in range(masks_np.shape[0]):

            y_true = masks_np[i]
            y_pred = preds_np[i]

            total_patches += 1

            if (y_true > 0).sum() > 0:
                gt_damage_patches += 1

                iou = compute_damage_iou(y_true, y_pred)
                if iou is not None:
                    damage_only_ious.append(iou)

    # -----------------------------------------------------
    # Final metrics
    # -----------------------------------------------------
    results = {
        "model": "dual_branch_diff_gated_v3",
        "mean_iou": float(mean_iou_metric.result().numpy()),
        "per_class_iou": {
            "background": float(per_class_0.result().numpy()),
            "minor": float(per_class_1.result().numpy()),
            "major": float(per_class_2.result().numpy()),
            "destroyed": float(per_class_3.result().numpy()),
        },
        "damage_only_stats": {
            "total_patches": total_patches,
            "gt_damage_patches": gt_damage_patches,
            "percentage_gt_damage":
                float(100.0 * gt_damage_patches / total_patches),
            "mean_damage_iou_gt_only":
                float(np.mean(damage_only_ious))
                if damage_only_ious else None
        }
    }

    # -----------------------------------------------------
    # Print
    # -----------------------------------------------------
    print("\n======================================")
    print("FINAL Test Evaluation (GATED DIFF v3)")
    print("======================================")
    print(json.dumps(results, indent=4))
    print("======================================\n")

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------
    output_path = os.path.join(experiment_dir, "test_metrics.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"[INFO] Saved to: {output_path}\n")


# =========================================================
# Entry
# =========================================================
if __name__ == "__main__":
    main()
