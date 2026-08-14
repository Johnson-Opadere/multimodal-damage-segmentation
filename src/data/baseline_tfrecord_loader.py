"""
TFRecord Dataset Loader for Multimodal Disaster Damage Segmentation.

Overview:
---------
This module provides utilities for loading TFRecord datasets containing
multimodal inputs (RGB + SAR) and segmentation masks.

It is designed to:
- Efficiently parse serialized TFRecord examples
- Reconstruct image tensors and masks
- Build high-performance tf.data pipelines

Input Data:
-----------
TFRecords are expected to contain:

    rgb_pre   : (512,512,3) float32
    rgb_post  : (512,512,3) float32
    sar_pre   : (512,512,8) float32
    sar_post  : (512,512,8) float32
    damage_mask      : (512,512) uint8
    sar_channel_mask : (8,) uint8

Output:
--------
- image: (512,512,22) float32
- mask : (512,512) int32

Pipeline Features:
------------------
- Parallel TFRecord reading
- Optional shuffling
- Deterministic or non-deterministic mapping
- Batching
- Prefetching for performance

Run Example:
------------
from src.data.tfrecord_loader import get_dataset

train_ds = get_dataset(
    tfrecord_dir="tfrecords/train/balanced",
    batch_size=4,
    shuffle=True,
    repeat=True
)
"""

import tensorflow as tf

# ---------------------------------------------------
# Constants
# ---------------------------------------------------

IMAGE_HEIGHT = 512
IMAGE_WIDTH  = 512

RGB_CHANNELS = 3
SAR_CHANNELS = 8

TOTAL_CHANNELS = 22  # 3+3+8+8


# ---------------------------------------------------
# Feature Description
# ---------------------------------------------------

def _get_feature_description():
    """
    Define TFRecord feature schema.

    Returns:
        Dict[str, tf.io.FixedLenFeature]:
            Mapping of feature names to TFRecord parsing definitions.

    Features:
    ---------
    - patch_id (string)
    - rgb_pre / rgb_post (serialized float32)
    - sar_pre / sar_post (serialized float32)
    - damage_mask (serialized uint8)
    - sar_channel_mask (serialized uint8)
    - has_damage (int64)
    - has_destroyed (int64)

    Notes:
        - All image-like tensors are stored as raw byte strings
        - Shapes are reconstructed during parsing
    """
    return {
        "patch_id": tf.io.FixedLenFeature([], tf.string),

        "rgb_pre": tf.io.FixedLenFeature([], tf.string),
        "rgb_post": tf.io.FixedLenFeature([], tf.string),

        "sar_pre": tf.io.FixedLenFeature([], tf.string),
        "sar_post": tf.io.FixedLenFeature([], tf.string),

        "damage_mask": tf.io.FixedLenFeature([], tf.string),
        "sar_channel_mask": tf.io.FixedLenFeature([], tf.string),

        "has_damage": tf.io.FixedLenFeature([], tf.int64),
        "has_destroyed": tf.io.FixedLenFeature([], tf.int64),
    }


# ---------------------------------------------------
# Parse Function
# ---------------------------------------------------

def _parse_example(example_proto):
    """
    Parse a single TFRecord example into tensors.

    Args:
        example_proto (tf.Tensor): Serialized TFRecord example

    Returns:
        Tuple:
            image (tf.Tensor): (512,512,22) float32
            mask  (tf.Tensor): (512,512) int32

    Pipeline:
    ---------
    1. Parse serialized example using feature schema
    2. Decode raw byte strings into tensors
    3. Reshape tensors to original dimensions
    4. Concatenate modalities into a single image tensor

    Image Construction:
    -------------------
    image = concat(
        rgb_pre  (3 channels),
        rgb_post (3 channels),
        sar_pre  (8 channels),
        sar_post (8 channels)
    ) → total 22 channels

    Mask Processing:
    ----------------
    - Loaded as uint8
    - Cast to int32 for training compatibility

    Notes:
        - No augmentation applied here
        - Assumes correct serialization format
    """
    feature_description = _get_feature_description()
    parsed = tf.io.parse_single_example(example_proto, feature_description)

    # ---- Decode ----
    rgb_pre  = tf.io.decode_raw(parsed["rgb_pre"], tf.float32)
    rgb_post = tf.io.decode_raw(parsed["rgb_post"], tf.float32)
    sar_pre  = tf.io.decode_raw(parsed["sar_pre"], tf.float32)
    sar_post = tf.io.decode_raw(parsed["sar_post"], tf.float32)

    mask = tf.io.decode_raw(parsed["damage_mask"], tf.uint8)

    # ---- Reshape ----
    rgb_pre  = tf.reshape(rgb_pre,  [IMAGE_HEIGHT, IMAGE_WIDTH, RGB_CHANNELS])
    rgb_post = tf.reshape(rgb_post, [IMAGE_HEIGHT, IMAGE_WIDTH, RGB_CHANNELS])

    sar_pre  = tf.reshape(sar_pre,  [IMAGE_HEIGHT, IMAGE_WIDTH, SAR_CHANNELS])
    sar_post = tf.reshape(sar_post, [IMAGE_HEIGHT, IMAGE_WIDTH, SAR_CHANNELS])

    mask = tf.reshape(mask, [IMAGE_HEIGHT, IMAGE_WIDTH])
    mask = tf.cast(mask, tf.int32)

    # ---- Concatenate ----
    image = tf.concat(
        [rgb_pre, rgb_post, sar_pre, sar_post],
        axis=-1
    )

    return image, mask


# ---------------------------------------------------
# Dataset Builder
# ---------------------------------------------------

def build_dataset(
    tfrecord_paths,
    batch_size,
    shuffle=False,
    repeat=False,
    deterministic=True
):
    """
    Build a tf.data.Dataset pipeline from TFRecord files.

    Args:
        tfrecord_paths (List[str]):
            List of TFRecord shard file paths
        batch_size (int):
            Number of samples per batch
        shuffle (bool):
            Whether to shuffle dataset (default: False)
        repeat (bool):
            Whether to repeat dataset indefinitely (default: False)
        deterministic (bool):
            Whether to enforce deterministic mapping order

    Returns:
        tf.data.Dataset:
            Batched dataset yielding (image, mask)

    Pipeline Steps:
    ---------------
    1. Load TFRecordDataset (parallel reads)
    2. Optional shuffle
    3. Parse examples (_parse_example)
    4. Optional repeat
    5. Batch dataset
    6. Prefetch for performance

    Performance Features:
    ---------------------
    - AUTOTUNE parallel reads
    - AUTOTUNE map calls
    - Prefetching to overlap CPU/GPU work

    Notes:
        - Deterministic=False improves performance but reduces reproducibility
        - Shuffle buffer size is fixed at 128
    """

    dataset = tf.data.TFRecordDataset(
        tfrecord_paths,
        num_parallel_reads=tf.data.AUTOTUNE
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=128,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        _parse_example,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=deterministic
    )

    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# ---------------------------------------------------
# Public Dataset API
# ---------------------------------------------------

def get_dataset(
    tfrecord_dir,
    batch_size,
    shuffle=False,
    repeat=False,
    deterministic=True
):
    """
    Load TFRecord dataset from directory (convenience wrapper).

    Args:
        tfrecord_dir (str):
            Directory containing TFRecord shards (*.tfrecord)
        batch_size (int):
            Batch size for training/inference
        shuffle (bool):
            Whether to shuffle dataset
        repeat (bool):
            Whether to repeat dataset indefinitely
        deterministic (bool):
            Whether to enforce deterministic execution

    Returns:
        tf.data.Dataset:
            Ready-to-use dataset pipeline

    Workflow:
    ---------
    1. Discover TFRecord files using glob
    2. Validate file existence
    3. Build dataset pipeline using build_dataset()

    Raises:
        ValueError:
            If no TFRecord files are found in directory

    Run Example:
    ------------
    train_ds = get_dataset(
        tfrecord_dir="tfrecords/train/balanced",
        batch_size=4,
        shuffle=True,
        repeat=True
    )

    Notes:
        - Designed for easy integration into training scripts
        - Supports train/val/test splits
    """

    tfrecord_paths = tf.io.gfile.glob(f"{tfrecord_dir}/*.tfrecord")

    if not tfrecord_paths:
        raise ValueError(f"No TFRecord files found in {tfrecord_dir}")

    return build_dataset(
        tfrecord_paths=tfrecord_paths,
        batch_size=batch_size,
        shuffle=shuffle,
        repeat=repeat,
        deterministic=deterministic
    )