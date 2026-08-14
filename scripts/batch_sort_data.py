#!/usr/bin/env python3
"""
batch_sort_data.py
=================

Batch preprocessing script for Project 1 dataset.

Pipeline:
---------
1. Identify invalid events from labels
2. Apply filtering + organization to:
   - labels
   - images
   - targets

Usage:
------
python scripts/batch_sort_data.py

Run Example:
------------
export DATA_ROOT=/mnt/ebs-data/cv_project1_new/data
python scripts/batch_sort_data.py

"""

import os
import logging
from src.preprocessing.separate_by_postfix import (
    separate_by_postfix,
    get_invalid_files
)


# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# -------------------------------------------------
# Config
# -------------------------------------------------
BASE_PATH = os.getenv("DATA_ROOT", "./data")

SPLITS = ["train", "hold", "test"]
DATA_TYPES = ["labels", "images", "targets"]


# -------------------------------------------------
# Core Pipeline
# -------------------------------------------------
def process_split(split: str):
    """
    Process a dataset split (train/hold/test)
    """
    logging.info(f"Processing split: {split}")

    labels_path = os.path.join(BASE_PATH, split, "labels")

    invalid_files = get_invalid_files(labels_path)
    logging.info(f"{split} invalid files: {len(invalid_files)}")

    for data_type in DATA_TYPES:
        folder_path = os.path.join(BASE_PATH, split, data_type)

        logging.info(f"Processing {data_type}...")
        separate_by_postfix(
            folder_path,
            invalid_files)


def run():
    for split in SPLITS:
        process_split(split)


if __name__ == "__main__":
    run()