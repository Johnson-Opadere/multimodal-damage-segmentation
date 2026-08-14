#!/usr/bin/env python3
"""
Damage Mask Generation Pipeline

This module converts polygon annotations (GeoJSON) into dense segmentation masks.

Pipeline Overview:
------------------
1. Load pixel-space annotation polygons (0–1023)
2. Scale geometries to match model resolution (512×512)
3. Rasterize polygons into dense masks
4. Save masks as NumPy arrays

Used in:
--------
Project 1 — Disaster Damage Segmentation

Output:
-------
- Dense segmentation masks (512×512)
- Classes: {0,1,2,3}

Run Example:
------------
export DATA_ROOT=/mnt/ebs-data/cv_project1_new/data
python src/annotations/generate_damage_masks.py
"""

import os
import json
import numpy as np
from shapely.geometry import shape
from shapely.affinity import scale as scale_geom
from rasterio.features import rasterize
from tqdm import tqdm


# ========================
# Config
# ========================

TARGET_SIZE = (512, 512)
SCALE_FACTOR = 0.5  # 1024 → 512

DAMAGE_MAPPING = {
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
}


# ========================
# Geometry loading
# ========================

def load_and_scale_geometries(geojson_path):
    """
    Load and preprocess annotation geometries.

    Steps:
    ------
    1. Read GeoJSON file
    2. Extract polygon geometries
    3. Map damage labels to class IDs
    4. Remove invalid or empty geometries
    5. Scale coordinates from 1024×1024 → 512×512

    Args:
        geojson_path (str): Path to GeoJSON annotation file

    Returns:
        List[Tuple[shapely.geometry, int]]:
            List of (geometry, class_id) tuples

    Notes:
        - Background / no-damage polygons are skipped
        - Scaling is required to match model input resolution
        - Assumes pixel coordinate system (no CRS)
    """
    with open(geojson_path, "r") as f:
        data = json.load(f)

    shapes = []

    for feat in data.get("features", []):
        props = feat.get("properties", {})
        subtype = str(props.get("subtype", "")).lower()
        class_id = DAMAGE_MAPPING.get(subtype, 0)

        if class_id == 0:
            continue

        geom = shape(feat["geometry"])
        if geom.is_empty or not geom.is_valid:
            continue

        geom_scaled = scale_geom(
            geom,
            xfact=SCALE_FACTOR,
            yfact=SCALE_FACTOR,
            origin=(0, 0)
        )

        shapes.append((geom_scaled, class_id))

    return shapes


# ========================
# Rasterization
# ========================

def rasterize_mask(shapes, out_path):
    """
    Convert polygon geometries into a dense segmentation mask.

    Steps:
    ------
    1. Rasterize polygons into 2D mask
    2. Assign class IDs to pixels
    3. Enforce dtype and shape constraints
    4. Save mask as .npy file

    Args:
        shapes (List[Tuple[geometry, int]]): Scaled geometries with class labels
        out_path (str): Output file path (.npy)

    Returns:
        np.ndarray:
            Mask of shape (512, 512), dtype uint8

    Notes:
        - Empty inputs produce all-zero masks
        - Ensures strict uint8 output (no float leakage)
        - Background = 0
    """
    if not shapes:
        mask = np.zeros(TARGET_SIZE, dtype=np.uint8)
    else:
        mask = rasterize(
            shapes=shapes,
            out_shape=TARGET_SIZE,
            fill=0,
            dtype=np.uint8
        )

    mask = np.asarray(mask, dtype=np.uint8)

    assert mask.dtype == np.uint8, "Mask dtype is not uint8"
    assert mask.shape == TARGET_SIZE, "Mask shape mismatch"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, mask)

    return mask


# ========================
# Split-level processing
# ========================

def generate_masks_for_split(split_dir, split_name):
    """
    Generate segmentation masks for a dataset split.

    Workflow:
    ---------
    1. Load annotation GeoJSON files
    2. Convert polygons to pixel-space masks
    3. Save masks to disk

    Args:
        split_dir (str): Path to dataset split directory
        split_name (str): Name of split (train, hold, test)

    Outputs:
        - mask/*.npy files for each annotation

    Notes:
        - Only post-disaster annotations are used
        - Skips missing or invalid files
        - Logs progress and errors
    """
    annot_dir = os.path.join(split_dir, "annot_post")
    out_dir = os.path.join(split_dir, "mask")

    if not os.path.exists(annot_dir):
        print(f"[WARN] annot_post not found for {split_name}")
        return

    files = [
        f for f in os.listdir(annot_dir)
        if f.lower().endswith((".geojson", ".json"))
    ]

    if not files:
        print(f"[WARN] No annotation files in {annot_dir}")
        return

    print(f"\n=== Generating damage masks for {split_name} ===")

    generated = 0
    for fname in tqdm(files, desc=f"{split_name} masks"):
        in_path = os.path.join(annot_dir, fname)
        base = os.path.splitext(fname)[0]
        out_path = os.path.join(out_dir, f"{base}_mask.npy")

        try:
            shapes = load_and_scale_geometries(in_path)
            rasterize_mask(shapes, out_path)
            generated += 1
        except Exception as e:
            print(f"[ERROR] {fname}: {e}")

    print(f"[OK] {generated}/{len(files)} masks generated for {split_name}")


# ========================
# Main
# ========================

if __name__ == "__main__":
    """
    Entry point for damage mask generation.

    Workflow:
    ---------
    - Iterate over dataset splits (train, hold, test)
    - Generate masks for each split
    - Save results to dataset_root/<split>/mask/

    Notes:
        - Assumes dataset is already preprocessed
        - Requires annotation GeoJSON files

    Run Example:
    ------------
    export DATA_ROOT=/mnt/ebs-data/cv_project1_new/data
    python src/annotations/generate_damage_masks.py
    """

    DATA_ROOT = os.getenv("DATA_ROOT", ".")
    dataset_root = os.path.join(DATA_ROOT, "rgb_data")

    splits = ["train", "hold", "test"]

    for split in splits:
        split_dir = os.path.join(dataset_root, split)
        if not os.path.exists(split_dir):
            print(f"[WARN] Split not found: {split}")
            continue

        generate_masks_for_split(split_dir, split)

    print("\n Damage mask generation complete.")