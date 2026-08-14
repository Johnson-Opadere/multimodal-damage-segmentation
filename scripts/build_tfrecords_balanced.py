#!/usr/bin/env python3
"""
Build BALANCED TFRecords for Project 1 (Multimodal Damage Segmentation).

Overview:
---------
This script constructs a class-balanced TFRecord dataset from preprocessed
multimodal inputs (RGB, SAR, masks). Balancing is applied ONLY to the training set.

Balancing Strategy:
-------------------
Each patch is categorized based on damage distribution:

Category A — No damage
    fg_pixels == 0

Category B — Damage present but no destroyed
    fg_pixels > 0 AND destroyed_pixels == 0

Category C — Destroyed present
    destroyed_pixels > 0

Target Quotas (for 2100 training samples):
------------------------------------------
A: 30% → 630 samples
B: 40% → 840 samples
C: 30% → 630 samples

Design Guarantees:
------------------
- Deterministic patch ordering (sorted filenames)
- Deterministic quota selection (no randomness)
- Same schema as RAW TFRecords
- Fixed shard size (SHARD_SIZE)
- Strict dtype enforcement:
    RGB/SAR → float32
    mask    → uint8
- Raw byte serialization (no compression)

Expected Input Structure:
--------------------------
DATA_ROOT/
    normalized_data/
        train/
    mask/
        train/

Output Structure:
-----------------
OUTPUT_ROOT/
    train/balanced/
        train_balanced_0000.tfrecord
        train_balanced_0001.tfrecord
        ...

Notes:
------
- Only TRAIN split is used
- Excess samples beyond quota are discarded deterministically
- Failed patch loads are skipped

Run Example:
------------
export DATA_ROOT=/mnt/ebs-data/cv_project1_new/data
export OUTPUT_ROOT=/mnt/ebs-data/cv_project1_new/tfrecords_v2

python scripts/build_tfrecords_balanced.py
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
TRAIN_SPLIT = "train"

# Balanced quotas (absolute counts)
TARGET_A = 630
TARGET_B = 840
TARGET_C = 630

# ============================================================
# TF Feature Helpers
# ============================================================

def _bytes_feature(value: bytes):
    """
    Create a TensorFlow bytes Feature.

    Args:
        value (bytes): Serialized byte content

    Returns:
        tf.train.Feature: BytesList feature

    Notes:
        - Used for storing raw arrays (RGB, SAR, masks)
        - Arrays must be converted via `.tobytes()`
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value: int):
    """
    Create a TensorFlow int64 Feature.

    Args:
        value (int): Integer value

    Returns:
        tf.train.Feature: Int64List feature

    Notes:
        - Used for metadata flags (e.g., has_damage)
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# ============================================================
# Serialization
# ============================================================

def build_example(
    patch_id,
    rgb_pre,
    rgb_post,
    sar_pre,
    sar_post,
    damage_mask,
    sar_channel_mask,
):

    """
    Serialize a single patch into a TFRecord Example.

    Args:
        patch_id (str): Unique patch identifier
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
    - Serializes arrays as raw bytes
    - Adds metadata flags:
        has_damage     → any(mask > 0)
        has_destroyed  → any(mask == 3)

    Notes:
        - Schema is identical to RAW TFRecords
        - No compression applied
    """

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
# Patch Loading
# ============================================================

def load_patch(base_name):
    """
    Load all modalities for a training patch.

    Args:
        base_name (str): Patch identifier

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
    normalized_data/train/
        rgb_pre_norm/
        rgb_post_norm/
        sar_pre_norm/
        sar_post_norm/

    mask/train/
        *_post_disaster_mask.npy

    Notes:
        - Only TRAIN split is used
        - Assumes strict naming conventions
        - Raises exception if any file is missing
    """

    split_root = os.path.join(DATA_ROOT)

    rgb_pre = np.load(
        os.path.join(split_root, "normalized_data", "train",
                     "rgb_pre_norm", f"{base_name}_pre_disaster_norm.npy")
    )
    rgb_post = np.load(
        os.path.join(split_root, "normalized_data", "train",
                     "rgb_post_norm", f"{base_name}_post_disaster_norm.npy")
    )

    sar_pre = np.load(
        os.path.join(split_root, "normalized_data", "train",
                     "sar_pre_norm", f"{base_name}_pre_sar_norm.npy")
    )
    sar_post = np.load(
        os.path.join(split_root, "normalized_data", "train",
                     "sar_post_norm", f"{base_name}_post_sar_norm.npy")
    )

    sar_channel_mask = np.load(
        os.path.join(split_root, "normalized_data", "train",
                     "sar_post_norm", f"{base_name}_post_sar_channel_mask.npy")
    )

    damage_mask = np.load(
        os.path.join(split_root, "mask", "train",
                     f"{base_name}_post_disaster_mask.npy")
    )

    return rgb_pre, rgb_post, sar_pre, sar_post, damage_mask, sar_channel_mask

# ============================================================
# Balanced Builder
# ============================================================

def build_balanced_train():
    """
    Build balanced TFRecords for the training dataset.

    Workflow:
    ---------
    1. Load all patch IDs from mask directory
    2. Categorize patches into A/B/C based on damage distribution
    3. Select deterministic subsets using predefined quotas
    4. Serialize selected patches into TFRecord Examples
    5. Write fixed-size shards to disk

    Categories:
    -----------
    A → No damage (background only)
    B → Damage present, no destroyed
    C → Destroyed present

    Output:
    -------
    OUTPUT_ROOT/train/balanced/
        train_balanced_0000.tfrecord
        train_balanced_0001.tfrecord
        ...

    Notes:
        - Selection is deterministic (no randomness)
        - Ensures reproducibility
        - Improves class balance for training
        - Failed loads are skipped with logging
    """

    print("\n=== Building BALANCED TFRecords (train only) ===")

    mask_dir = os.path.join(DATA_ROOT, "mask", "train")

    files = sorted(
        f for f in os.listdir(mask_dir)
        if f.endswith("_post_disaster_mask.npy")
    )

    patch_ids = [
        f.replace("_post_disaster_mask.npy", "")
        for f in files
    ]

    print(f"Total train patches: {len(patch_ids)}")

    # ---------------------------------------------------------
    # Categorize
    # ---------------------------------------------------------

    cat_A = []
    cat_B = []
    cat_C = []

    for pid in tqdm(patch_ids, desc="Categorizing"):

        mask = np.load(
            os.path.join(mask_dir, f"{pid}_post_disaster_mask.npy")
        )

        fg_pixels = np.sum(mask > 0)
        destroyed_pixels = np.sum(mask == 3)

        if fg_pixels == 0:
            cat_A.append(pid)
        elif destroyed_pixels == 0:
            cat_B.append(pid)
        else:
            cat_C.append(pid)

    print("\nCategory counts (available):")
    print(f"A (no damage)       : {len(cat_A)}")
    print(f"B (damage no dest)  : {len(cat_B)}")
    print(f"C (destroyed)       : {len(cat_C)}")

    # ---------------------------------------------------------
    # Deterministic quota selection
    # ---------------------------------------------------------

    selected = (
        cat_A[:TARGET_A] +
        cat_B[:TARGET_B] +
        cat_C[:TARGET_C]
    )

    print("\nSelected for balanced:")
    print(f"A: {len(cat_A[:TARGET_A])}")
    print(f"B: {len(cat_B[:TARGET_B])}")
    print(f"C: {len(cat_C[:TARGET_C])}")
    print(f"TOTAL: {len(selected)}")

    # ---------------------------------------------------------
    # Write shards
    # ---------------------------------------------------------

    output_dir = os.path.join(OUTPUT_ROOT, "train", "balanced")
    os.makedirs(output_dir, exist_ok=True)

    shard_buffer = []
    shard_index = 0

    for pid in tqdm(selected, desc="Writing shards"):

        try:
            data = load_patch(pid)
        except Exception as e:
            print(f"[ERROR] Loading {pid}: {e}")
            continue

        serialized = build_example(pid, *data)
        shard_buffer.append(serialized)

        if len(shard_buffer) == SHARD_SIZE:

            shard_path = os.path.join(
                output_dir,
                f"train_balanced_{shard_index:04d}.tfrecord"
            )

            with tf.io.TFRecordWriter(shard_path) as writer:
                for record in shard_buffer:
                    writer.write(record)

            shard_buffer = []
            shard_index += 1

    if shard_buffer:
        shard_path = os.path.join(
            output_dir,
            f"train_balanced_{shard_index:04d}.tfrecord"
        )

        with tf.io.TFRecordWriter(shard_path) as writer:
            for record in shard_buffer:
                writer.write(record)

    print("\n[OK] Balanced TFRecords built successfully.")

# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    """
    Entry point for building BALANCED TFRecords.

    Executes:
        - build_balanced_train()

    Run Example:
    ------------
    export DATA_ROOT=/mnt/ebs-data/cv_project1_new/data
    export OUTPUT_ROOT=/mnt/ebs-data/cv_project1_new/tfrecords_v2

    python scripts/build_tfrecords_balanced.py

    Output:
    -------
    Balanced TFRecord shards written to OUTPUT_ROOT/train/balanced/
    """

    build_balanced_train()

    print("\n BALANCED TFRecord build complete.")


