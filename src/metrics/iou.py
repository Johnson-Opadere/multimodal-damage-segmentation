# src/metrics/iou.py
"""
IoU Metrics for Multiclass Segmentation.

Overview:
---------
This module provides a wrapper around TensorFlow's MeanIoU metric
to support softmax-based model outputs in segmentation tasks.

Problem:
--------
tf.keras.metrics.MeanIoU expects:
    - y_pred as integer class labels

However, segmentation models output:
    - y_pred as softmax probabilities (B, H, W, C)

Solution:
---------
This wrapper converts probabilities to class labels using argmax
before computing IoU.

Use Case:
---------
- Multiclass segmentation (e.g., disaster damage classification)
- Compatible with softmax outputs from neural networks

Classes:
--------
0 → Background  
1 → Minor damage  
2 → Major damage  
3 → Destroyed  

Run Example:
------------
from src.metrics.iou import MeanIoUWrapper

metric = MeanIoUWrapper(num_classes=4)

model.compile(
    optimizer="adam",
    loss=loss_fn,
    metrics=[metric]
)
"""

import tensorflow as tf


class MeanIoUWrapper(tf.keras.metrics.MeanIoU):
    """
    Wrapper for MeanIoU metric to handle softmax predictions.

    This class extends tf.keras.metrics.MeanIoU by automatically
    converting predicted probability maps into discrete class labels
    using argmax before computing the metric.

    Args:
        num_classes (int):
            Number of segmentation classes
        name (str):
            Name of the metric (default: "mean_iou")
        **kwargs:
            Additional arguments passed to base class

    Behavior:
    ---------
    - Accepts:
        y_true → (B, H, W) integer labels
        y_pred → (B, H, W, C) softmax probabilities

    - Converts:
        y_pred → argmax(y_pred, axis=-1)

    - Computes:
        Standard Mean Intersection-over-Union

    Notes:
        - Required for compatibility with softmax outputs
        - Avoids manual argmax preprocessing in training loop
    """
    def __init__(self, num_classes, name="mean_iou", **kwargs):
        super().__init__(num_classes=num_classes, name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)
