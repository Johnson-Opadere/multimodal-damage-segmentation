#!/usr/bin/env python3
# scripts/preprocess_rgb.py (Project 2, enforce (512,512) resize)

"""
RGB Preprocessing Pipeline

This module handles preprocessing of RGB satellite imagery:
-----------------------------------------------------------
- Resize images to fixed resolution (512×512)
- Normalize using ImageNet statistics
- Save processed arrays as NumPy (.npy)
- Optional visualization for debugging

Used in:
--------
Project 1 / Project 2 — RGB preprocessing for multimodal learning

Run Example:
------------
export DATA_ROOT=/mnt/ebs-data/cv_project1_new/data
python scripts/preprocess_rgb.py
"""

import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# ImageNet mean and std for normalization (RGB order, [0-1] scale)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TARGET_SIZE = (512, 512)  # enforce fixed resolution


def normalize_rgb(img_arr: np.ndarray) -> np.ndarray:
    """
    Normalize RGB image using ImageNet statistics.

    Args:
        img_arr (np.ndarray): Input image array (H, W, 3), uint8, range [0, 255]

    Returns:
        np.ndarray: Normalized image (float32, H, W, 3)

    Notes:
        - Converts to [0,1] scale before normalization
        - Standard normalization used for pretrained CNNs (e.g., ResNet)
    """
    img_arr = img_arr.astype(np.float32) / 255.0
    return (img_arr - IMAGENET_MEAN) / IMAGENET_STD


def process_rgb_png(in_path: str, out_path: str):
    """
    Process a single RGB PNG image.

    Steps:
    ------
    1. Load image
    2. Convert to RGB
    3. Resize to 512×512
    4. Normalize using ImageNet stats
    5. Save as NumPy (.npy)

    Args:
        in_path (str): Input image path (.png)
        out_path (str): Output path (.npy)

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Raw resized image (uint8)
            - Normalized image (float32)
    """
    img = Image.open(in_path).convert("RGB")

    # enforce 512×512
    img = img.resize(TARGET_SIZE, Image.BILINEAR)

    arr = np.array(img, dtype=np.uint8)
    arr_norm = normalize_rgb(arr)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, arr_norm.astype(np.float32))

    print(f"[OK] {in_path} -> {out_path} | shape={arr_norm.shape}")
    return arr, arr_norm


def show_visualization(raw_arr, norm_arr, title="RGB Patch", save_dir=None):
    """
    Visualize raw and normalized RGB images.

    Displays:
    ---------
    - Raw RGB image
    - Normalized image (rescaled for visualization)

    Args:
        raw_arr (np.ndarray): Raw image (uint8)
        norm_arr (np.ndarray): Normalized image
        title (str): Plot title
        save_dir (str, optional): Directory to save visualization

    Notes:
        - Normalized image is rescaled to [0,1] for display only
        - Uses percentile clipping to handle outliers
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(title, fontsize=12)

    axes[0].imshow(raw_arr)
    axes[0].set_title("Raw RGB")
    axes[0].axis("off")

    disp = np.zeros_like(norm_arr)
    for c in range(3):
        if np.all(np.isnan(norm_arr[:, :, c])):
            disp[:, :, c] = 0.5
        else:
            vmin, vmax = np.nanpercentile(norm_arr[:, :, c], [2, 98])
            disp[:, :, c] = np.clip((norm_arr[:, :, c] - vmin) / (vmax - vmin + 1e-6), 0, 1)

    axes[1].imshow(disp)
    axes[1].set_title("Normalized (visualized)")
    axes[1].axis("off")

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{title.replace(os.sep,'_')}.png")
        plt.savefig(out_path, dpi=150)
        print(f"[VIS] Saved visualization: {out_path}")

    plt.show()


def batch_process_rgb(in_dir: str, out_dir: str, visualize: bool = False, n_samples: int = 2, vis_dir=None):
    """
    Batch process RGB images in a directory.

    Args:
        in_dir (str): Input directory containing PNG images
        out_dir (str): Output directory for processed .npy files
        visualize (bool): Whether to visualize sample images
        n_samples (int): Number of samples to visualize
        vis_dir (str): Directory to save visualization outputs

    Notes:
        - Processes all .png files in directory
        - Preserves naming convention for pre/post disaster images
        - Skips files with unexpected naming
    """
    files = [f for f in os.listdir(in_dir) if f.endswith(".png")]

    if visualize and files:
        sample_files = random.sample(files, min(n_samples, len(files)))
    else:
        sample_files = []

    for idx, fname in enumerate(files, start=1):
        in_path = os.path.join(in_dir, fname)

        if "_pre_disaster" in fname:
            base_name = fname.replace(".png", "_norm.npy")
        elif "_post_disaster" in fname:
            base_name = fname.replace(".png", "_norm.npy")
        else:
            print(f"[WARN] Unexpected filename format: {fname}")
            continue

        out_path = os.path.join(out_dir, base_name)

        print(f"[{idx}/{len(files)}] Processing {fname}")
        raw_arr, arr_norm = process_rgb_png(in_path, out_path)

        if fname in sample_files:
            show_visualization(raw_arr, arr_norm, title=fname, save_dir=vis_dir)


def run_all():
    """
    Run RGB preprocessing across dataset splits.

    Workflow:
    ---------
    For each split (train, hold, test):
        For each RGB type (pre, post):
            1. Load images
            2. Resize + normalize
            3. Save processed arrays
            4. Optionally visualize samples

    Notes:
        - Assumes dataset structure:
            dataset_root/
                ├── train/
                ├── hold/
                ├── test/
        - Output stored in *_norm directories

    Run Example:
        export DATA_ROOT=/mnt/ebs-data/cv_project1_new/data
        python scripts/preprocess_rgb.py
    """

    #dataset_root = r"C:/Users/Johns/.../rgb_data"
    DATA_ROOT = os.getenv("DATA_ROOT", ".")
    dataset_root = os.path.join(DATA_ROOT, "rgb_data")

    vis_root = os.path.join(dataset_root, "debug_vis", "rgb")

    splits = ["train", "hold", "test"]
    rgb_types = ["rgb_pre", "rgb_post"]

    for split in splits:
        for rgb_type in rgb_types:
            in_dir = os.path.join(dataset_root, split, rgb_type)
            out_dir = os.path.join(dataset_root, split, rgb_type + "_norm")
            vis_dir = os.path.join(vis_root, split, rgb_type)

            print(f"\n[INFO] Processing {in_dir} -> {out_dir}")
            os.makedirs(out_dir, exist_ok=True)

            batch_process_rgb(in_dir, out_dir, visualize=True, n_samples=2, vis_dir=vis_dir)


if __name__ == "__main__":
    run_all()