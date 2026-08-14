# src/metrics/per_class_iou.py
"""
Per-Class IoU Metric for Multiclass Segmentation.

Overview:
---------
This module implements a per-class Intersection-over-Union (IoU) metric
for detailed evaluation of segmentation performance.

Unlike MeanIoU, which aggregates performance across all classes,
this metric computes IoU for a specific class, enabling:

- Fine-grained analysis of class-wise performance
- Better debugging of class imbalance issues
- Targeted improvements (e.g., destroyed vs minor damage)

Use Case:
---------
- Multiclass segmentation with imbalanced classes
- Monitoring performance of critical classes (e.g., "destroyed")

Classes:
--------
0 → Background  
1 → Minor damage  
2 → Major damage  
3 → Destroyed  

Run Example:
------------
from src.metrics.per_class_iou import PerClassIoU

metrics = [
    PerClassIoU(num_classes=4, class_id=0),
    PerClassIoU(num_classes=4, class_id=1),
    PerClassIoU(num_classes=4, class_id=2),
    PerClassIoU(num_classes=4, class_id=3),
]

model.compile(
    optimizer="adam",
    loss=loss_fn,
    metrics=metrics
)
"""

import tensorflow as tf


class PerClassIoU(tf.keras.metrics.Metric):
    """
    Compute IoU for a specific class using a confusion matrix.

    Args:
        num_classes (int):
            Total number of segmentation classes
        class_id (int):
            Target class for IoU computation
        name (str, optional):
            Metric name (default: "iou_class_<class_id>")
        **kwargs:
            Additional arguments for base Metric class

    Internal State:
    ---------------
    confusion_matrix (tf.Variable):
        Accumulates counts across batches:
            shape = (num_classes, num_classes)

    Behavior:
    ---------
    - Accepts:
        y_true → (B, H, W) integer labels
        y_pred → (B, H, W, C) softmax probabilities

    - Converts:
        y_pred → argmax(y_pred)

    - Updates:
        Confusion matrix over all batches

    - Computes:
        IoU for specified class:
            IoU = TP / (TP + FP + FN)

    Notes:
        - Designed for detailed evaluation and debugging
        - Complements MeanIoU for class-level insights
    """
    def __init__(self, num_classes, class_id, name=None, **kwargs):
        super().__init__(name=name or f"iou_class_{class_id}", **kwargs)
        self.num_classes = num_classes
        self.class_id = class_id

        self.confusion_matrix = self.add_weight(
            name="conf_matrix",
            shape=(num_classes, num_classes),
            initializer="zeros",
            dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update confusion matrix with a new batch.

        Args:
            y_true (tf.Tensor):
                Ground truth labels, shape (B, H, W)
            y_pred (tf.Tensor):
                Predicted probabilities, shape (B, H, W, C)
            sample_weight (tf.Tensor, optional):
                Not used (included for API compatibility)

        Processing:
        -----------
        1. Convert softmax predictions to class labels:
            y_pred = argmax(y_pred)

        2. Flatten tensors:
            → shape (-1,)

        3. Compute batch confusion matrix

        4. Accumulate into global confusion matrix

        Notes:
            - Maintains running statistics across batches
            - Enables dataset-level IoU computation
        """

        y_pred = tf.argmax(y_pred, axis=-1)

        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        cm = tf.math.confusion_matrix(
            y_true,
            y_pred,
            num_classes=self.num_classes,
            dtype=tf.float32
        )

        self.confusion_matrix.assign_add(cm)

    def result(self):
        """
        Update confusion matrix with a new batch.

        Args:
            y_true (tf.Tensor):
                Ground truth labels, shape (B, H, W)
            y_pred (tf.Tensor):
                Predicted probabilities, shape (B, H, W, C)
            sample_weight (tf.Tensor, optional):
                Not used (included for API compatibility)

        Processing:
        -----------
        1. Convert softmax predictions to class labels:
            y_pred = argmax(y_pred)

        2. Flatten tensors:
            → shape (-1,)

        3. Compute batch confusion matrix

        4. Accumulate into global confusion matrix

        Notes:
            - Maintains running statistics across batches
            - Enables dataset-level IoU computation
        """

        cm = self.confusion_matrix

        tp = cm[self.class_id, self.class_id]
        fp = tf.reduce_sum(cm[:, self.class_id]) - tp
        fn = tf.reduce_sum(cm[self.class_id, :]) - tp

        denominator = tp + fp + fn

        return tf.math.divide_no_nan(tp, denominator)

    def reset_states(self):
        """
        Reset confusion matrix to initial state.

        Behavior:
        ---------
        - Clears accumulated statistics
        - Called automatically at start of each epoch

        Notes:
            - Ensures metrics are computed per-epoch
        """
        self.confusion_matrix.assign(
            tf.zeros_like(self.confusion_matrix)
        )
