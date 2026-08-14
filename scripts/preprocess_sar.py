#!/usr/bin/env python3
"""
SAR preprocessing (Project 2 & Project 1 compatible)

This script preprocesses Sentinel-1 SAR imagery exported in dB space.

Normalization strategy:
  • Linear scaling in a canonical dB range (NOT per-image z-score)
  • Canonical Sentinel-1 dB range clipping [-30, +5]
  • Preserves absolute backscatter physics and inter-scene comparability
  • Robust to low-variance SAR patches

Features:
  • Linear scaling in canonical dB space
  • Explicit nodata handling (-9999)
  • Resize to 512×512 (after NaN removal)
  • Pad to fixed 8 SAR channels
  • Explicit SAR channel mask (1 = real, 0 = padded)
  • Strict pre/post pairing
  • Defensive assertion against value explosions
  • Sanity statistics

Outputs:
  sar_pre_norm/*.npy                    → (512,512,8)
  sar_pre_norm/*_sar_channel_mask.npy  → (8,)
  sar_post_norm/*.npy                   → (512,512,8)
  sar_post_norm/*_sar_channel_mask.npy → (8,)

Run Example:
------------
export DATA_ROOT=/mnt/ebs-data/cv_project1_new/data
python src/scripts/preprocess_sar.py
"""

import os
import glob
import numpy as np
import rasterio
from tqdm import tqdm
from collections import defaultdict
from skimage.transform import resize

# ========================
# Config
# ========================
#dataset_root = r"C:/Users/Johns/OneDrive/Desktop/CV Projects/Project 2/dataset_root"

# ✅ ONLY PATH FIX
DATA_ROOT = os.getenv("DATA_ROOT", ".")
dataset_root = os.path.join(DATA_ROOT, "sar_data")

splits = ["train", "hold", "test"]

TARGET_SIZE = (512, 512)
MAX_SAR_CHANNELS = 8            # VV+VH × 4 dates (max)
SAR_NODATA = -9999.0            # MUST match SAR downloader
SAR_ABS_MAX_ALLOWED = 2.0       # After scaling, values should be ~[-1, 0.2]

# Canonical Sentinel-1 dB range
SAR_DB_MIN = -30.0
SAR_DB_MAX = 5.0
SAR_DB_SCALE = abs(SAR_DB_MIN)  # = 30.0

# ========================
# Stats collectors
# ========================
stats = defaultdict(list)
processed_counter = 0
skipped_counter = 0
empty_counter = 0

# ========================
# Helpers
# ========================
def normalize_sar_db(band: np.ndarray, nodata: float | None) -> np.ndarray:
    """
    Normalize a single SAR band using linear scaling in canonical dB space.

    Processing steps:
    -----------------
    1. Replace nodata values with NaN
    2. Clip values to Sentinel-1 dB range [-30, +5]
    3. Scale linearly by dividing by 30.0
    4. Replace NaNs with zeros

    Args:
        band (np.ndarray): Input SAR band (H, W) in dB space
        nodata (float | None): Nodata value (typically -9999)

    Returns:
        np.ndarray: Normalized SAR band (float32)

    Guarantees:
        - No NaN or Inf values in output
        - Stable numerical range
        - Preserves physical meaning of SAR backscatter

    Notes:
        - Avoids per-image normalization (important for consistency across scenes)
        - Suitable for multi-scene and multi-event training
    """
    band = band.astype(np.float32)

    if nodata is not None:
        band = np.where(band == nodata, np.nan, band)

    if np.all(np.isnan(band)):
        return np.zeros_like(band, dtype=np.float32)

    band = np.clip(band, SAR_DB_MIN, SAR_DB_MAX)
    band = band / SAR_DB_SCALE
    band = np.nan_to_num(band, nan=0.0)

    return band


# (REST OF FILE UNCHANGED — EXACTLY AS YOUR ORIGINAL)