# src/losses/combined_loss.py
"""
Combined Loss Functions for Multiclass Damage Segmentation.

Overview:
---------
This module implements loss functions used for training the segmentation model,
with a focus on handling class imbalance in disaster damage datasets.

The primary loss used is a combination of:
- Sparse Categorical Crossentropy (CCE)
- Weighted Soft Dice Loss

Motivation:
-----------
The dataset is highly imbalanced:
- Background dominates
- Minor/Major/Destroyed classes are underrepresented

Using CCE alone:
    → biased toward background

Using Dice alone:
    → unstable early training

Solution:
---------
Combine both:

    Loss = α * CCE + β * Dice

Where:
- CCE → stabilizes optimization
- Dice → improves overlap and minority class learning

Classes:
--------
0 → Background  
1 → Minor damage  
2 → Major damage  
3 → Destroyed  

Run Example:
------------
from src.losses.combined_loss import combined_cce_dice_loss

model.compile(
    optimizer="adam",
    loss=combined_cce_dice_loss(),
    metrics=["accuracy"]
)
"""

import tensorflow as tf


# -----------------------------------------------------
# Weighted Dice Loss (multi-class, soft)
# -----------------------------------------------------

def weighted_dice_loss(
    y_true,
    y_pred,    
    class_weights = (0.05, 0.45, 0.30, 0.20),
    smooth=1e-6
):
    """
    Multi-class weighted soft Dice loss.

    Args:
        y_true (tf.Tensor):
            Ground truth labels, shape (B, H, W), dtype int32
        y_pred (tf.Tensor):
            Predicted probabilities (softmax), shape (B, H, W, C)
        class_weights (Tuple[float]):
            Per-class weights for Dice contribution
        smooth (float):
            Smoothing factor to prevent division by zero

    Returns:
        tf.Tensor: Scalar Dice loss

    Pipeline:
    ---------
    1. Convert y_true to one-hot encoding
    2. Flatten spatial dimensions
    3. Compute per-class Dice score:
        Dice = (2 * intersection + smooth) / (union + smooth)
    4. Apply class weights
    5. Normalize weighted Dice
    6. Return (1 - Dice)

    Class Weights:
    --------------
    Current setting:
        (0.05, 0.45, 0.30, 0.20)

    Interpretation:
        - Background heavily down-weighted
        - Minor damage emphasized
        - Major/destroyed balanced

    Notes:
        - Soft Dice (uses probabilities, not argmax)
        - Handles severe class imbalance
        - Historical weight configs retained for experiment tracking
    """

    num_classes = tf.shape(y_pred)[-1]

    # Convert y_true to one-hot
    y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)
    y_true_onehot = tf.cast(y_true_onehot, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Flatten spatial dims
    y_true_flat = tf.reshape(y_true_onehot, [-1, num_classes])
    y_pred_flat = tf.reshape(y_pred, [-1, num_classes])

    intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=0)
    denominator = tf.reduce_sum(y_true_flat + y_pred_flat, axis=0)

    dice = (2.0 * intersection + smooth) / (denominator + smooth)

    # Apply class weights
    weights = tf.constant(class_weights, dtype=tf.float32)
    weighted_dice = dice * weights

    # Normalize by sum of weights to keep scale stable
    weighted_mean_dice = tf.reduce_sum(weighted_dice) / tf.reduce_sum(weights)

    return 1.0 - weighted_mean_dice


# -----------------------------------------------------
# Combined Loss
# -----------------------------------------------------

def combined_cce_dice_loss(cce_weight=0.4, dice_weight=0.6):
    """
    Combined Sparse Categorical Crossentropy + Weighted Dice Loss.

    Args:
        cce_weight (float):
            Weight for cross-entropy component (default: 0.4)
        dice_weight (float):
            Weight for Dice loss component (default: 0.6)

    Returns:
        Callable:
            Loss function (y_true, y_pred) → scalar loss

    Formula:
    --------
        Loss = cce_weight * CCE + dice_weight * Dice

    Components:
    -----------
    - SparseCategoricalCrossentropy:
        • Pixel-wise classification loss
        • Stable gradients early in training

    - Weighted Dice Loss:
        • Overlap-based metric
        • Improves segmentation quality
        • Handles class imbalance

    Design Choice:
    --------------
    Dice is weighted higher (0.6) because:
        - Dataset is highly imbalanced
        - Overlap quality is more critical than raw accuracy

    Notes:
        - Suitable for multi-class segmentation
        - Works with softmax outputs
        - Balances stability (CCE) and performance (Dice)

    Run Example:
    ------------
    loss_fn = combined_cce_dice_loss()

    model.compile(
        optimizer="adam",
        loss=loss_fn,
        metrics=["accuracy"]
    )
    """

    cce = tf.keras.losses.SparseCategoricalCrossentropy()

    def loss_fn(y_true, y_pred):
        cce_loss = cce(y_true, y_pred)
        d_loss = weighted_dice_loss(y_true, y_pred)
        return cce_weight * cce_loss + dice_weight * d_loss

    return loss_fn