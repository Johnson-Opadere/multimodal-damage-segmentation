#!/usr/bin/env python3
"""
Dual-Branch GATED DIFF v2 Training Script (Context-Aware Gating)

Overview:
---------
This script trains the upgraded segmentation model incorporating:

- Dual-branch temporal architecture (PRE / POST)
- Explicit temporal difference modeling (DIFF)
- Context-aware gated fusion (v2)
- Adaptive multiscale training pipeline (v2i)

This represents a key architectural upgrade over both:
    - Baseline (no DIFF)
    - DIFF (no gating)
    - GATED DIFF v1 (pixel-wise gating)

Model:
------
Dual-Branch GATED DIFF v2:

    PRE branch  = RGB_pre + SAR_pre
    POST branch = RGB_post + SAR_post

Temporal modeling:
    RGB_diff = RGB_post - RGB_pre
    SAR_diff = SAR_post - SAR_pre

Fusion:
    Context-aware gated DIFF applied at feature level

Key Innovation:
---------------
v1 gating:
    gate = sigmoid(conv1x1(diff))

v2 gating (context-aware):
    gate = sigmoid(conv3x3 → conv3x3 → conv1x1(diff))

Benefits:
---------
- Captures spatial context
- Reduces SAR noise sensitivity
- Improves damage-class separation
- Stabilizes training (via BatchNorm on DIFF)

Data:
-----
TFRecords (correct SAR):

    tfrecords_v2/
        train/balanced/
        val/raw/

Data Pipeline:
--------------
Adaptive multiscale sampling (v2i):
    - 50% full image
    - 30% medium crop (384 → resized)
    - 20% small crop (256 → resized)

Damage-aware cropping priority:
    1. destroyed
    2. general damage
    3. random fallback

Loss:
-----
Combined CCE + Dice (balanced)

Metrics:
--------
- Accuracy
- MeanIoU
- Per-class IoU

Expected Performance:
---------------------
Baseline (~0.495)
GATED v1 (~0.497)
GATED v2 (~0.500–0.503)

Outputs:
--------
Saved to:
    experiments_v2/dual_branch_diff_gated_v2_<timestamp>/

Includes:
    - best_model.keras
    - training_log.csv
    - TensorBoard logs

Run Example:
------------
PYTHONPATH=. python scripts_v2/train_dual_branch_diff_gated_v2.py
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


from src.models.dual_branch_diff_gated_segmentation_model_v2 import build_segmentation_model
from src.data.multiscale_adaptive_damage_crop_loader import get_dataset

from src.losses.combined_loss import combined_cce_dice_loss
from src.metrics.iou import MeanIoUWrapper
from src.metrics.per_class_iou import PerClassIoU


# =====================================================
# Learning Rate Logger
# =====================================================

class LearningRateLogger(tf.keras.callbacks.Callback):
    """
    Custom callback to log learning rate at the end of each epoch.

    Purpose:
    --------
    Tracks the evolution of learning rate during training,
    especially when using adaptive schedulers such as ReduceLROnPlateau.

    Behavior:
    ---------
    - Retrieves current optimizer learning rate
    - Adds it to logs under key "lr"

    Benefits:
    ---------
    - Enables correlation between LR changes and performance
    - Stored in CSV logs and TensorBoard

    Notes:
        - Useful for diagnosing plateau or instability issues
    """
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["lr"] = float(
            tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        )


# =====================================================
# Main
# =====================================================

def main():
    """
    Main training pipeline for GATED DIFF v2 model.

    Workflow:
    ---------
    1. Initialize experiment directory
    2. Load adaptive multiscale datasets
    3. Build GATED DIFF v2 model
    4. Compile with loss and metrics
    5. Configure callbacks
    6. Train model
    7. Save outputs

    Steps:
    ------

    [1] Experiment Setup
        - Create timestamped directory
        - Ensures reproducibility and experiment tracking

    [2] Dataset Loading
        - Train: balanced TFRecords + adaptive multiscale sampling
        - Validation: raw TFRecords (no augmentation)

    [3] Model Initialization
        - Dual-branch architecture
        - Context-aware gated DIFF fusion
        - Full training (no freezing)

    [4] Compilation
        - Optimizer: Adam (1e-5)
        - Loss: Combined CCE + Dice
        - Metrics:
            • Accuracy
            • MeanIoU
            • Per-class IoU (all classes)

    [5] Callbacks
        - ModelCheckpoint:
            Save best model (val_mean_iou)
        - ReduceLROnPlateau:
            Adaptive learning rate reduction
        - EarlyStopping:
            Prevent overfitting
        - LearningRateLogger:
            Log LR per epoch
        - CSVLogger:
            Save training history
        - TensorBoard:
            Enable visualization

    [6] Training
        - Train for up to 60 epochs
        - Monitor validation IoU for early stopping

    Notes:
    ------
    - Only architectural change vs DIFF script is gating mechanism
    - Data pipeline and loss remain constant for fair comparison
    - Designed for final performance push (~0.50 mIoU threshold)

    Output:
    -------
    All artifacts saved to experiment directory
    """

    # -------------------------------------------------
    # Experiment Directory
    # -------------------------------------------------
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    run_dir = f"experiments_v2/dual_branch_diff_gated_v2_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n[INFO] Logging to: {run_dir}\n")


    # -------------------------------------------------
    # Dataset (IDENTICAL — DO NOT TOUCH)
    # -------------------------------------------------
    print("[INFO] Building adaptive multiscale dataset...")

    train_dataset = get_dataset(
        tfrecord_dir="tfrecords_v2/train/balanced",
        batch_size=4,
        shuffle=True
    )

    val_dataset = get_dataset(
        tfrecord_dir="tfrecords_v2/val/raw",
        batch_size=4,
        shuffle=False
    )


    # -------------------------------------------------
    # Model (GATED DIFF v2)
    # -------------------------------------------------
    print("[INFO] Building GATED DIFF v2 model...")

    model = build_segmentation_model(
        input_shape=(512, 512, 22),
        num_classes=4,
        freeze_early=False
    )


    # -------------------------------------------------
    # Loss (IDENTICAL)
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
    # Callbacks (IDENTICAL)
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
            patience=10,
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
            histogram_freq=0
        )
    ]


    # -------------------------------------------------
    # Train (ONLY CHANGE: epochs = 60)
    # -------------------------------------------------
    print("\n[INFO] Starting training (GATED DIFF_v2)...\n")

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=60,   #  UPDATED
        callbacks=callbacks
    )

    print("\n[INFO] Training complete.")


# =====================================================
# Entry
# =====================================================

if __name__ == "__main__":
    """
    Entry point for training script.

    Executes:
        main()

    Usage:
    ------
    PYTHONPATH=. python scripts_v2/train_dual_branch_diff_gated_v2.py

    Notes:
        - Ensure virtual environment is activated
        - GPU memory growth is configured before execution
    """
    main()