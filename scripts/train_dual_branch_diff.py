#!/usr/bin/env python3
"""
Dual-Branch DIFF Training Script (Multiscale)

Overview:
---------
This script trains the Dual-Branch + DIFF segmentation model using the
best-performing training configuration (v2i), combining:

- Explicit temporal difference modeling (DIFF)
- Adaptive multiscale cropping
- Balanced TFRecord dataset
- Combined loss (CCE + Dice)
- Full metric tracking and checkpointing

Model:
------
Dual-Branch DIFF Architecture:
    - PRE branch: RGB_pre + SAR_pre + DIFF
    - POST branch: RGB_post + SAR_post + DIFF

Where:
    RGB_diff = RGB_post - RGB_pre
    SAR_diff = SAR_post - SAR_pre

Data:
-----
TFRecords (corrected SAR):

    tfrecords_v2/
        train/balanced/   (training)
        val/raw/          (validation)

Data Pipeline:
--------------
Adaptive Multiscale Sampling (v2i):
    - 50% → full image
    - 30% → medium crop (384 → resized)
    - 20% → small crop (256 → resized)

    Damage-aware cropping prioritizes:
        1. destroyed class
        2. general damage
        3. random fallback

Loss:
-----
Combined CCE + Dice:
    - Balanced optimization
    - Improved segmentation overlap

Metrics:
--------
- Accuracy
- MeanIoU
- Per-class IoU (all 4 classes)

Purpose:
--------
Evaluate performance gain from explicit temporal modeling:

    Baseline → implicit change learning
    DIFF     → explicit change injection

Expected Improvement:
---------------------
+0.005 → +0.01 mIoU

Outputs:
--------
Saved to:
    experiments_v2/dual_branch_diff_<timestamp>/

Includes:
    - best_model.keras
    - training_log.csv
    - TensorBoard logs

Run Example:
------------
PYTHONPATH=. python scripts_v2/train_dual_branch_diff.py
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

#  KEY CHANGE: DIFF MODEL
from src.models.dual_branch_diff_segmentation_model import build_segmentation_model

#  SAME AS v2i
from src.data.multiscale_adaptive_damage_crop_loader import get_dataset

from src.losses.combined_loss import combined_cce_dice_loss
from src.metrics.iou import MeanIoUWrapper
from src.metrics.per_class_iou import PerClassIoU


# =====================================================
# Learning Rate Logger
# =====================================================

class LearningRateLogger(tf.keras.callbacks.Callback):
    """
    Custom callback to log learning rate at each epoch.

    Purpose:
    --------
    Tracks dynamic learning rate changes during training,
    especially when using schedulers such as ReduceLROnPlateau.

    Behavior:
    ---------
    - Extracts current optimizer learning rate
    - Logs it under key "lr"

    Benefits:
    ---------
    - Enables analysis of LR vs performance
    - Stored in CSV logs and TensorBoard

    Notes:
        - Helps debug convergence behavior
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
    Main training pipeline for Dual-Branch DIFF model (v2i).

    Workflow:
    ---------
    1. Create timestamped experiment directory
    2. Load adaptive multiscale datasets
    3. Build DIFF segmentation model
    4. Compile with combined loss and metrics
    5. Configure callbacks
    6. Train model
    7. Save outputs

    Steps:
    ------

    [1] Experiment Setup
        - Create unique run directory
        - Ensures reproducibility and traceability

    [2] Dataset Loading
        - Train: balanced TFRecords + adaptive multiscale sampling
        - Validation: raw TFRecords (no augmentation)

    [3] Model Initialization
        - Dual-branch architecture with DIFF injection
        - Full training (no frozen layers)

    [4] Compilation
        - Optimizer: Adam (1e-5)
        - Loss: Combined CCE + Dice
        - Metrics:
            • Accuracy
            • MeanIoU
            • Per-class IoU

    [5] Callbacks
        - ModelCheckpoint:
            Save best model based on validation IoU
        - ReduceLROnPlateau:
            Adaptive learning rate scheduling
        - EarlyStopping:
            Prevent overfitting
        - LearningRateLogger:
            Track learning rate changes
        - CSVLogger:
            Save training history
        - TensorBoard:
            Enable visualization

    [6] Training
        - Train for up to 50 epochs
        - Monitor validation IoU

    Notes:
    ------
    - Uses best-performing pipeline (v2i)
    - Combines strong data + model improvements
    - Designed for final performance evaluation

    Output:
    -------
    All artifacts saved to experiment directory for analysis
    """

    # -------------------------------------------------
    # Experiment Directory
    # -------------------------------------------------
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    run_dir = f"experiments_v2/dual_branch_diff_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n[INFO] Logging to: {run_dir}\n")


    # -------------------------------------------------
    # Dataset (ADAPTIVE MULTISCALE — SAME AS v2i)
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
    # Model (DIFF)
    # -------------------------------------------------
    print("[INFO] Building DIFF model...")

    model = build_segmentation_model(
        input_shape=(512, 512, 22),
        num_classes=4,
        freeze_early=False
    )


    # -------------------------------------------------
    # Loss
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
    print("\n[INFO] Starting training (DIFF + v2i)...\n")

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
    """
    Entry point for training script.

    Executes:
        main()

    Usage:
    ------
    PYTHONPATH=. python scripts_v2/train_dual_branch_diff.py

    Notes:
        - Ensure environment is activated
        - GPU memory growth is configured before model initialization
    """
    main()