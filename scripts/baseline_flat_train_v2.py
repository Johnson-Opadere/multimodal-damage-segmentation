#!/usr/bin/env python3
"""
Baseline Training Script (SAR Corrected)

Overview:
---------
This script trains the baseline segmentation model using corrected SAR data,
where temporal information is properly preserved (SAR_pre ≠ SAR_post).

It serves as a controlled re-run of the original baseline experiment to
quantify the impact of fixing the SAR temporal bug.

Model:
------
Dual-Branch ResNet50 + UNet Decoder + Flat 4-Class Softmax

- Separate PRE and POST branches
- Multimodal fusion (RGB + SAR)
- Pixel-wise segmentation output

Data:
-----
TFRecords (corrected SAR):

    tfrecords_v2/
        train/balanced/
        val/raw/

Loss:
-----
Balanced Tversky Loss:
    - Handles class imbalance
    - Emphasizes damage classes over background

Metrics:
--------
- Accuracy
- MeanIoU
- Per-class IoU (background, minor, major, destroyed)

Experiment Purpose:
-------------------
A/B comparison:

    Baseline (bugged SAR)  vs  Baseline (correct SAR)

Ensures:
- Architecture remains identical
- Only data changes (causal attribution)

Outputs:
--------
Saved to:
    experiments_v2/baseline_flat_correct_<timestamp>/

Includes:
    - best_model.keras
    - training_log.csv
    - TensorBoard logs

Run Example:
------------
source venv/bin/activate
python scripts_v2/baseline_flat_train_v2.py

Optional:
nohup python scripts_v2/baseline_flat_train_v2.py > logs/baseline_v2.log 2>&1 &
"""

import os
import datetime
import tensorflow as tf

# =====================================================
# GPU Memory Growth (MUST RUN BEFORE MODEL INIT)
# =====================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled.")
    except RuntimeError as e:
        print(e)

# =====================================================
# Imports — Explicit and Traceable
# =====================================================
from src.models.baseline_segmentation_model import build_segmentation_model
from src.data.baseline_tfrecord_loader import get_dataset
from src.losses.balanced_tversky_loss import balanced_tversky_loss
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
    Ensures learning rate is recorded in:
        - CSV logs
        - TensorBoard logs

    This is important when using:
        - ReduceLROnPlateau
        - Dynamic learning rate schedules

    Behavior:
    ---------
    - Extracts current optimizer learning rate
    - Injects it into logs dictionary as "lr"

    Notes:
        - Useful for debugging convergence behavior
        - Helps correlate performance with LR changes
    """
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["lr"] = float(
            tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        )


# =====================================================
# Main Training Function
# =====================================================
def main():
    """
    Main training pipeline.

    Workflow:
    ---------
    1. Create versioned experiment directory
    2. Load TFRecord datasets (corrected SAR)
    3. Build baseline segmentation model
    4. Compile with optimizer, loss, and metrics
    5. Configure training callbacks
    6. Train model
    7. Save outputs (model, logs, TensorBoard)

    Steps:
    ------

    [1] Experiment Setup
        - Create timestamped run directory
        - Ensures reproducibility and isolation

    [2] Dataset Loading
        - Train: balanced dataset (class-balanced)
        - Validation: raw dataset (true distribution)

    [3] Model Initialization
        - Dual-branch ResNet50 encoder
        - UNet-style decoder
        - 4-class softmax output

    [4] Compilation
        - Optimizer: Adam (1e-5)
        - Loss: Balanced Tversky
        - Metrics:
            • Accuracy
            • MeanIoU
            • Per-class IoU

    [5] Callbacks
        - ModelCheckpoint:
            Save best model (val_mean_iou)
        - ReduceLROnPlateau:
            Adaptive LR reduction
        - EarlyStopping:
            Prevent overfitting
        - LearningRateLogger:
            Track LR evolution
        - CSVLogger:
            Save training history
        - TensorBoard:
            Visual monitoring

    [6] Training
        - Train for fixed epochs (25)
        - Monitor validation performance

    Notes:
    ------
    - Designed for controlled SAR bug fix experiment
    - Keeps architecture constant for fair comparison
    - Uses balanced training data for improved learning

    Output:
    -------
    All artifacts saved to experiment directory for analysis
    """

    # -------------------------------------------------
    # Experiment Directory (v2)
    # -------------------------------------------------
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = f"experiments_v2/baseline_flat_correct_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n[INFO] Logging to: {run_dir}\n")

    # -------------------------------------------------
    # Dataset (Correct SAR)
    # -------------------------------------------------
    print("[INFO] Loading datasets (tfrecords_v2)...")

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
    # Model — Baseline Architecture
    # -------------------------------------------------
    print("[INFO] Building model: Dual-Branch ResNet50 + UNet (Flat 4-Class)...")

    model = build_segmentation_model(
        input_shape=(512, 512, 22),
        num_classes=4,
        freeze_early=False  # full training
    )

    # -------------------------------------------------
    # Compile
    # -------------------------------------------------
    print("[INFO] Compiling model with Balanced Tversky loss...")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),

        loss=balanced_tversky_loss(
            alpha=0.2,
            beta=0.8,
            class_weights=[0.05, 0.35, 0.35, 0.25]
        ),

        metrics=[
            "accuracy",
            MeanIoUWrapper(num_classes=4),
            PerClassIoU(num_classes=4, class_id=0),  # background
            PerClassIoU(num_classes=4, class_id=1),  # minor
            PerClassIoU(num_classes=4, class_id=2),  # major
            PerClassIoU(num_classes=4, class_id=3),  # destroyed
        ]
    )

    # -------------------------------------------------
    # Callbacks
    # -------------------------------------------------
    print("[INFO] Setting up callbacks...")

    callbacks = [

        # Save best model based on validation IoU
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(run_dir, "best_model.keras"),
            monitor="val_mean_iou",
            mode="max",
            save_best_only=True,
            verbose=1
        ),

        # Reduce LR on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_mean_iou",
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),

        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor="val_mean_iou",
            mode="max",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),

        # Learning rate logging
        LearningRateLogger(),

        # CSV logging
        tf.keras.callbacks.CSVLogger(
            filename=os.path.join(run_dir, "training_log.csv"),
            append=False
        ),

        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(run_dir, "tensorboard"),
            histogram_freq=1
        )
    ]

    # -------------------------------------------------
    # Train
    # -------------------------------------------------
    print("\n[INFO] Starting training...\n")

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=25,
        callbacks=callbacks
    )

    print("\n[INFO] Training complete.")


# =====================================================
# Entry Point
# =====================================================
if __name__ == "__main__":
    """
    Entry point for training script.

    Executes:
        main()

    Usage:
    ------
    python scripts_v2/baseline_flat_train_v2.py

    Notes:
        - Ensure environment is activated
        - GPU memory growth is configured before execution
    """
    main()