#!/usr/bin/env python3
"""
Dual-Branch Efficient GATED DIFF v3 Training Script (FINAL)

Overview:
---------
This script trains the final production model for multimodal disaster damage
segmentation using an efficient gated DIFF architecture (v3).

It represents the culmination of iterative improvements over:

    Baseline → DIFF → GATED v1 → GATED v2 → Efficient GATED v3

Key Capabilities:
-----------------
 Multimodal fusion (RGB + SAR)  
 Temporal modeling (pre vs post)  
 Efficient gated DIFF fusion (v3)  
 Adaptive multiscale training pipeline (v2i)  
 Optimized training configuration (AdamW + higher LR)  

Model:
------
Dual-Branch Efficient GATED DIFF v3:

    PRE branch  = RGB_pre + SAR_pre
    POST branch = RGB_post + SAR_post

Temporal Signal:
    RGB_diff = RGB_post - RGB_pre  
    SAR_diff = SAR_post - SAR_pre  

Fusion:
    Efficient context-aware gated DIFF (v3):
        - Channel compression
        - Lightweight gating
        - Reduced memory footprint

Data Pipeline:
--------------
Uses SAME pipeline as v2 (critical for fair comparison):

    src.data.multiscale_adaptive_damage_crop_loader.get_dataset

Features:
    - Adaptive multiscale sampling
    - Damage-aware cropping
    - Balanced training distribution

Loss:
-----
Combined CCE + Dice (balanced)

Optimizer:
----------
AdamW:
    - learning_rate = 3e-4
    - weight_decay = 1e-4

Higher LR enabled by:
    - Stable gating design
    - Improved gradient flow

Metrics:
--------
- Accuracy
- MeanIoU
- Per-class IoU

Outputs:
--------
Saved to:
    experiments_v2/dual_branch_diff_gated_v3_<timestamp>/

Includes:
    - best_model.keras
    - final_model.keras
    - training_log.csv

Run Example:
------------
PYTHONPATH=. python scripts_v2/train_dual_branch_diff_gated_v3.py
"""

# =====================================================
# ENV
# =====================================================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import datetime
import json


# =====================================================
# GPU SETUP
# =====================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# =====================================================
# IMPORTS
# =====================================================

from src.models.dual_branch_diff_gated_segmentation_model_v3 import build_segmentation_model
from src.data.multiscale_adaptive_damage_crop_loader import get_dataset
from src.losses.combined_loss import combined_cce_dice_loss
from src.metrics.iou import MeanIoUWrapper
from src.metrics.per_class_iou import PerClassIoU


# =====================================================
# CONFIG
# =====================================================

CONFIG = {
    """
    Training configuration parameters.

    Fields:
    -------
    input_shape:
        Model input tensor shape (H, W, C)

    batch_size:
        Number of samples per batch
        Increased in v3 due to improved efficiency

    learning_rate:
        Initial learning rate for optimizer
        Higher than v2 due to improved stability

    epochs:
        Maximum number of training epochs
        EarlyStopping typically halts earlier

    Notes:
    ------
    - Values tuned for v3 architecture
    - Designed for faster convergence
    """
    "input_shape": (512, 512, 22),
    "batch_size": 4,                 # v3 can handle bigger batch
    "learning_rate": 3e-4,           # higher than v2
    "epochs": 40                     # with early stopping
}


# =====================================================
# EXPERIMENT DIR
# =====================================================

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
run_dir = f"experiments_v2/dual_branch_diff_gated_v3_{timestamp}"
os.makedirs(run_dir, exist_ok=True)

print(f"\n[INFO] Logging to: {run_dir}\n")


# =====================================================
# DATASET (IDENTICAL TO v2)
# =====================================================
"""
Dataset loading using adaptive multiscale pipeline.

Key Design:
-----------
- Uses SAME data pipeline as v2 to ensure:
    fair architectural comparison
    no confounding variables

Train:
------
- Balanced dataset
- Adaptive multiscale sampling
- Damage-aware cropping

Validation:
-----------
- Raw dataset (no augmentation)
- Reflects real-world distribution

Notes:
------
- Critical for controlled experimentation
"""

print("[INFO] Building dataset (same as v2)...")

train_dataset = get_dataset(
    tfrecord_dir="tfrecords_v2/train/balanced",
    batch_size=CONFIG["batch_size"],
    shuffle=True
)

val_dataset = get_dataset(
    tfrecord_dir="tfrecords_v2/val/raw",
    batch_size=CONFIG["batch_size"],
    shuffle=False
)


# =====================================================
# MODEL
# =====================================================
"""
Model initialization.

Architecture:
-------------
Dual-Branch Efficient GATED DIFF v3

Key Improvements over v2:
------------------------
- Reduced parameter count
- Lower memory footprint
- Efficient gating mechanism
- Maintains performance while improving scalability

Notes:
------
- No freezing (full training)
- Designed for final production performance
"""

print("[INFO] Building v3 model...")

model = build_segmentation_model(
    input_shape=CONFIG["input_shape"],
    num_classes=4
)


# =====================================================
# LOSS
# =====================================================
"""
Loss function: Combined CCE + Dice.

Design:
-------
- CCE → stabilizes training
- Dice → improves segmentation overlap
- Balanced weighting (0.5 / 0.5)

Purpose:
--------
- Handle class imbalance
- Improve damage region segmentation quality
"""

loss_fn = combined_cce_dice_loss(
    cce_weight=0.5,
    dice_weight=0.5
)


# =====================================================
# OPTIMIZER
# =====================================================
"""
Optimizer: AdamW

Configuration:
--------------
- learning_rate: higher than v2 (3e-4)
- weight_decay: 1e-4

Benefits:
---------
- Better generalization (weight decay)
- Faster convergence (higher LR)
- Stable training with v3 architecture

Notes:
------
- Enabled by improved gating stability
"""

optimizer = tf.keras.optimizers.AdamW(
    learning_rate=CONFIG["learning_rate"],
    weight_decay=1e-4
)


# =====================================================
# COMPILE
# =====================================================
"""
Model compilation.

Components:
-----------
- Optimizer: AdamW
- Loss: Combined CCE + Dice
- Metrics:
    • Accuracy
    • MeanIoU
    • Per-class IoU

Purpose:
--------
Provides comprehensive evaluation:
    - Overall performance (MeanIoU)
    - Class-level performance (PerClassIoU)
"""

print("[INFO] Compiling model...")

model.compile(
    optimizer=optimizer,
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


# =====================================================
# CALLBACKS
# =====================================================
"""
Model compilation.

Components:
-----------
- Optimizer: AdamW
- Loss: Combined CCE + Dice
- Metrics:
    • Accuracy
    • MeanIoU
    • Per-class IoU

Purpose:
--------
Provides comprehensive evaluation:
    - Overall performance (MeanIoU)
    - Class-level performance (PerClassIoU)
"""

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
        factor=0.3,
        patience=5,
        mode="max",
        min_lr=1e-6,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_mean_iou",
        patience=8,
        mode="max",
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.CSVLogger(
        os.path.join(run_dir, "training_log.csv")
    )
]


# =====================================================
# TRAIN
# =====================================================
"""
Training loop.

Behavior:
---------
- Train model on adaptive dataset
- Validate on raw dataset
- Monitor validation MeanIoU

Notes:
------
- EarlyStopping typically halts before max epochs
- Best model saved automatically
"""

print("\n[INFO] Starting training (v3)...\n")

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=CONFIG["epochs"],
    callbacks=callbacks
)


# =====================================================
# SAVE FINAL
# =====================================================
"""
Final model saving.

Outputs:
--------
- best_model.keras  → best validation performance
- final_model.keras → last epoch model

Purpose:
--------
- Preserve best checkpoint
- Enable further evaluation or deployment
"""

final_path = os.path.join(run_dir, "final_model.keras")
model.save(final_path)

print("\n[INFO] Training complete")
print(f"[INFO] Saved to: {run_dir}")