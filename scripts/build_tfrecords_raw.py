#!/usr/bin/env python3
"""
Build RAW TFRecords for Project 1 (Multimodal Damage Segmentation).

Overview:
---------
This script converts preprocessed multimodal data (RGB, SAR, masks)
into serialized TFRecord files for efficient training.

Each TFRecord example contains:
    - RGB_pre   : (512,512,3) float32
    - RGB_post  : (512,512,3) float32
    - SAR_pre   : (512,512,C) float32
    - SAR_post  : (512,512,C) float32
    - damage_mask        : (512,512) uint8
    - sar_channel_mask   : (C,) uint8
    - metadata flags     : has_damage, has_destroyed

Design Guarantees:
------------------
- Deterministic patch ordering (sorted filenames)
- Fixed shard size (SHARD_SIZE)
- Raw byte serialization (no compression)
- Strict dtype enforcement:
    RGB/SAR → float32
    masks   → uint8
- No class balancing (RAW dataset)

Expected Input Structure:
--------------------------
DATA_ROOT/
    normalized_data/
        train/
        hold/
        test/
    mask/
        train/
        hold/
        test/

Output Structure:
-----------------
OUTPUT_ROOT/
    train/raw/
    val/raw/
    test/raw/

Notes:
------
- "hold" split is mapped to "val" in output
- Each TFRecord shard contains SHARD_SIZE examples
- Remaining samples are flushed in final shard

Run Example:
------------
export DATA_ROOT=/mnt/ebs-data/cv_project1_new/data
export OUTPUT_ROOT=/mnt/ebs-data/cv_project1_new/tfrecords_v2

python scripts/build_tfrecords_raw.py
"""

import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = "/mnt/ebs-data/cv_project1_new/data"
#OUTPUT_ROOT = "/mnt/ebs-data/cv_project1_new/tfrecords"
OUTPUT_ROOT = "/mnt/ebs-data/cv_project1_new/tfrecords_v2"

SHARD_SIZE = 128
SPLITS = ["train", "hold", "test"]  # hold = val

# ============================================================
# TF Feature Helpers
# ============================================================

def _bytes_feature(value: bytes):
	"""
    Create a TensorFlow bytes Feature.

    Args:
        value (bytes): Raw serialized byte content

    Returns:
        tf.train.Feature: BytesList feature

    Notes:
        - Used for storing raw arrays (RGB, SAR, masks)
        - Arrays must be pre-serialized via `.tobytes()`
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value: int):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# ============================================================
# Example Builder
# ============================================================

def build_example(
    patch_id: str,
    rgb_pre,
    rgb_post,
    sar_pre,
    sar_post,
    damage_mask,
    sar_channel_mask,
):
    """
    Serialize a single multimodal patch into a TFRecord Example.

    Args:
        patch_id (str): Unique identifier for the patch
        rgb_pre (np.ndarray): Pre-disaster RGB image (512,512,3)
        rgb_post (np.ndarray): Post-disaster RGB image (512,512,3)
        sar_pre (np.ndarray): Pre-disaster SAR stack (512,512,C)
        sar_post (np.ndarray): Post-disaster SAR stack (512,512,C)
        damage_mask (np.ndarray): Segmentation mask (512,512)
        sar_channel_mask (np.ndarray): Valid SAR channel mask (C,)

    Returns:
        bytes: Serialized TFRecord Example

    Processing:
    -----------
    - Enforces dtype consistency:
        RGB/SAR → float32
        mask    → uint8
    - Serializes arrays using raw byte encoding
    - Adds metadata flags:
        has_damage     → any(mask > 0)
        has_destroyed  → any(mask == 3)

    Notes:
        - No compression applied
        - Compatible with downstream TFRecord loaders
    """

    # Enforce dtypes
    rgb_pre = rgb_pre.astype(np.float32)
    rgb_post = rgb_post.astype(np.float32)
    sar_pre = sar_pre.astype(np.float32)
    sar_post = sar_post.astype(np.float32)
    damage_mask = damage_mask.astype(np.uint8)
    sar_channel_mask = sar_channel_mask.astype(np.uint8)

    feature = {
        "patch_id": _bytes_feature(patch_id.encode("utf-8")),
        "rgb_pre": _bytes_feature(rgb_pre.tobytes()),
        "rgb_post": _bytes_feature(rgb_post.tobytes()),
        "sar_pre": _bytes_feature(sar_pre.tobytes()),
        "sar_post": _bytes_feature(sar_post.tobytes()),
        "damage_mask": _bytes_feature(damage_mask.tobytes()),
        "sar_channel_mask": _bytes_feature(sar_channel_mask.tobytes()),
        "has_damage": _int64_feature(int(np.any(damage_mask > 0))),
        "has_destroyed": _int64_feature(int(np.any(damage_mask == 3))),
    }

    example = tf.train.Example(
        features=tf.train.Features(feature=feature)
    )

    return example.SerializeToString()

# ============================================================
# Patch Loader
# ============================================================

def load_patch(split, base_name):
    """
    Load all modalities for a single patch from disk.

    Args:
        split (str): Dataset split ("train", "hold", "test")
        base_name (str): Patch identifier (event-level ID)

    Returns:
        Tuple:
            rgb_pre (np.ndarray)
            rgb_post (np.ndarray)
            sar_pre (np.ndarray)
            sar_post (np.ndarray)
            damage_mask (np.ndarray)
            sar_channel_mask (np.ndarray)

    File Mapping:
    -------------
    normalized_data/<split>/
        rgb_pre_norm/
        rgb_post_norm/
        sar_pre_norm/
        sar_post_norm/

    mask/<split>/
        *_post_disaster_mask.npy

    Notes:
        - Assumes strict naming convention
        - Raises exception if any file is missing
        - SAR channel mask is loaded from post-SAR directory
    """

    # Normalized data
    norm_root = os.path.join(DATA_ROOT, "normalized_data", split)

    rgb_pre_path = os.path.join(norm_root, "rgb_pre_norm",
                                f"{base_name}_pre_disaster_norm.npy")

    rgb_post_path = os.path.join(norm_root, "rgb_post_norm",
                                 f"{base_name}_post_disaster_norm.npy")

    sar_pre_path = os.path.join(norm_root, "sar_pre_norm",
                                f"{base_name}_pre_sar_norm.npy")

    sar_post_path = os.path.join(norm_root, "sar_post_norm",
                                 f"{base_name}_post_sar_norm.npy")

    sar_mask_path = os.path.join(norm_root, "sar_post_norm",
                                 f"{base_name}_post_sar_channel_mask.npy")

    # Damage mask
    damage_mask_path = os.path.join(
        DATA_ROOT, "mask", split,
        f"{base_name}_post_disaster_mask.npy"
    )

    rgb_pre = np.load(rgb_pre_path)
    rgb_post = np.load(rgb_post_path)
    sar_pre = np.load(sar_pre_path)
    sar_post = np.load(sar_post_path)
    sar_channel_mask = np.load(sar_mask_path)
    damage_mask = np.load(damage_mask_path)

    return rgb_pre, rgb_post, sar_pre, sar_post, damage_mask, sar_channel_mask

# ============================================================
# Split Builder
# ============================================================

def build_split(split):
	"""
    Build RAW TFRecords for a given dataset split.

    Args:
        split (str): Dataset split ("train", "hold", "test")

    Workflow:
    ---------
    1. Collect patch IDs from mask directory
    2. Sort for deterministic ordering
    3. Load multimodal data for each patch
    4. Serialize into TFRecord Examples
    5. Write fixed-size shards

    Output:
    -------
    OUTPUT_ROOT/<split>/raw/
        <split>_raw_0000.tfrecord
        <split>_raw_0001.tfrecord
        ...

    Special Handling:
    -----------------
    - "hold" → "val" (output naming)
    - Failed loads are skipped with error logging

    Notes:
        - Ensures reproducibility via sorted input
        - Uses SHARD_SIZE for batching
    """

    print(f"\n=== Building RAW TFRecords for {split} ===")

    out_split = "val" if split == "hold" else split

    output_dir = os.path.join(OUTPUT_ROOT, out_split, "raw")
    os.makedirs(output_dir, exist_ok=True)

    # Collect deterministic patch list
    mask_dir = os.path.join(DATA_ROOT, "mask", split)

    files = sorted(
        f for f in os.listdir(mask_dir)
        if f.endswith("_post_disaster_mask.npy")
    )

    patch_ids = [
        f.replace("_post_disaster_mask.npy", "")
        for f in files
    ]

    print(f"Total patches: {len(patch_ids)}")

    shard_buffer = []
    shard_index = 0

    for patch_id in tqdm(patch_ids):

        try:
            data = load_patch(split, patch_id)
        except Exception as e:
            print(f"[ERROR] Loading {patch_id}: {e}")
            continue

        serialized = build_example(patch_id, *data)
        shard_buffer.append(serialized)

        if len(shard_buffer) == SHARD_SIZE:

            shard_path = os.path.join(
                output_dir,
                f"{out_split}_raw_{shard_index:04d}.tfrecord"
            )

            with tf.io.TFRecordWriter(shard_path) as writer:
                for record in shard_buffer:
                    writer.write(record)

            shard_buffer = []
            shard_index += 1

    # Final flush
    if shard_buffer:
        shard_path = os.path.join(
            output_dir,
            f"{out_split}_raw_{shard_index:04d}.tfrecord"
        )

        with tf.io.TFRecordWriter(shard_path) as writer:
            for record in shard_buffer:
                writer.write(record)

    print(f"[OK] Finished {split}")

# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
	"""
    Entry point for TFRecord construction.

    Iterates through all dataset splits and builds RAW TFRecords.

    Splits:
        - train
        - hold → mapped to val
        - test

    Run Example:
    ------------
    export DATA_ROOT=/mnt/ebs-data/cv_project1_new/data
    export OUTPUT_ROOT=/mnt/ebs-data/cv_project1_new/tfrecords_v2

    python scripts/build_tfrecords_raw.py

    Output:
    -------
    TFRecord shards written to OUTPUT_ROOT with deterministic ordering.
    """

    for split in SPLITS:
        build_split(split)

    print("\n RAW TFRecord build complete.")