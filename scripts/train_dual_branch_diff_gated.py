#!/usr/bin/env python3
"""
================================================================================
Project 1 — Dual-Branch GATED DIFF Training Script (Multiscale v2i FINAL)
================================================================================

Purpose
-------
Train the FINAL model:

    ✔ Dual-branch (PRE / POST)
    ✔ Explicit DIFF signal
    ✔ Learnable gating on DIFF
    ✔ Adaptive multiscale training (v2i)

--------------------------------------------------------------------------------
Key Upgrade (CRITICAL)
--------------------------------------------------------------------------------

Previous DIFF model issue:
    DIFF dominates → overfitting → lower val IoU

Solution:
    Introduce GATING:

        diff = post - pre
        gate = sigmoid(conv(diff))

        output = fused(pre, post) + gate * diff

→ Model learns WHEN to trust DIFF

--------------------------------------------------------------------------------
Why This Matters
--------------------------------------------------------------------------------

Baseline:
    No temporal signal

DIFF:
    Strong signal, but overfits

GATED DIFF:
    Balanced signal → best generalization

--------------------------------------------------------------------------------
Expected Outcome
--------------------------------------------------------------------------------

Baseline (~0.495)
DIFF     (~0.48)
GATED    (~0.50+) 🎯

--------------------------------------------------------------------------------
Run
--------------------------------------------------------------------------------

PYTHONPATH=. python scripts_v2/train_dual_branch_diff_gated.py

================================================================================
"""

import os
import datetime
import tensorflow as tf


# =====================================================
# GPU Setup
# =====================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# =====================================================
# Imports
# =====================================================

# 🔥 FINAL MODEL
from src.models.dual_branch_diff_gated_segmentation_model import build_segmentation_model

# 🔥 SAME DATA PIPELINE (v2i)
from src.data.multiscale_adaptive_damage_crop_loader import get_dataset

from src.losses.combined_loss import combined_cce_dice_loss
from src.metrics.iou import MeanIoUWrapper
from src.metrics.per_class_iou import PerClassIoU


# =====================================================
# Learning Rate Logger
# =====================================================

class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["lr"] = float(
            tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        )


# =====================================================
# Main
# =====================================================

def main():

    # -------------------------------------------------
    # Experiment Directory
    # -------------------------------------------------
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    run_dir = f"experiments_v2/dual_branch_diff_gated_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n[INFO] Logging to: {run_dir}\n")


    # -------------------------------------------------
    # Dataset (IDENTICAL TO v2i — DO NOT CHANGE)
    # -------------------------------------------------
    print("[INFO] Building adaptive multiscale dataset...")

    train_dataset = get_dataset(
        tfrecord_dir="tfrecords_v2/train/balanced",
        batch_size=8,
        shuffle=True
    )

    val_dataset = get_dataset(
        tfrecord_dir="tfrecords_v2/val/raw",
        batch_size=8,
        shuffle=False
    )


    # -------------------------------------------------
    # Model (GATED DIFF)
    # -------------------------------------------------
    print("[INFO] Building GATED DIFF model...")

    model = build_segmentation_model(
        input_shape=(512, 512, 22),
        num_classes=4,
        freeze_early=False
    )


    # -------------------------------------------------
    # Loss (IDENTICAL — DO NOT CHANGE)
    # -------------------------------------------------
    print("[INFO] Compiling model...")

    loss_fn = combined_cce_dice_loss(
        cce_weight=0.5,
        dice_weight=0.5
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=loss_fn,
        metrics=[
            "accuracy",
            MeanIoUWrapper(num_classes=4),
            PerClassIoU(num_classes=4, class_id=0),
            PerClassIoU(num_classes=4, class_id=1),
            PerClassIoU(num_classes=4, class_id=2),
            PerClassIoU(num_classes=4, class_id=3),
        ]
    )


    # -------------------------------------------------
    # Callbacks (IDENTICAL TO v2i)
    # -------------------------------------------------
    print("[INFO] Setting up callbacks...")

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(run_dir, "best_model.keras"),
            monitor="val_mean_iou",
            mode="max",
            save_best_only=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_mean_iou",
            mode="max",
            factor=0.5,
            patience=6,
            min_lr=1e-6,
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_mean_iou",
            mode="max",
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),

        LearningRateLogger(),

        tf.keras.callbacks.CSVLogger(
            filename=os.path.join(run_dir, "training_log.csv"),
            append=False
        ),

        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(run_dir, "tensorboard"),
            histogram_freq=1
        )
    ]


    # -------------------------------------------------
    # Train
    # -------------------------------------------------
    print("\n[INFO] Starting training (GATED DIFF + v2i)...\n")

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,
        callbacks=callbacks
    )

    print("\n[INFO] Training complete.")


# =====================================================
# Entry
# =====================================================

if __name__ == "__main__":
    main()
