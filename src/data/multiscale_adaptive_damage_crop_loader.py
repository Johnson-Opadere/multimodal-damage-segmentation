"""
Multiscale Adaptive Damage Crop TFRecord Loader 

Overview:
---------
This module extends the standard TFRecord dataset loader with
adaptive multiscale cropping focused on damage regions.

It introduces:
- Damage-aware cropping with class prioritization
- Multiscale sampling (full / medium / small)
- Explicit emphasis on rare "destroyed" class

Goal:
-----
Improve model learning by:
- Increasing exposure to damaged regions
- Preventing under-sampling of rare classes
- Providing scale diversity during training

Key Enhancement:
----------------
Destroyed-aware cropping priority:

    1. destroyed pixels (class = 3)
    2. any damage (mask > 0)
    3. random fallback (no damage present)

This ensures:
- Rare "destroyed" class is consistently sampled
- Better class balance at patch level

Multiscale Strategy:
--------------------
- 50% → full image (512×512)
- 30% → medium crop (384×384 → resized to 512)
- 20% → small crop (256×256 → resized to 512)

Input:
------
TFRecords containing:
    image: (512,512,22)
    mask : (512,512)

Output:
-------
image: (512,512,22) float32  
mask : (512,512) int32  

Run Example:
------------
from src.data.multiscale_loader import get_dataset

train_ds = get_dataset(
    tfrecord_dir="tfrecords/train/balanced",
    batch_size=8,
    shuffle=True
)
"""

import tensorflow as tf
from src.data.tfrecord_loader import _parse_example

IMAGE_SIZE = 512
CROP_SMALL = 256
CROP_MEDIUM = 384


# =========================================================
# GENERIC DAMAGE-AWARE CROP (WITH DESTROYED PRIORITY)
# =========================================================
def _damage_aware_crop(image, mask, crop_size):
    """
    Perform damage-aware cropping with class prioritization.

    Args:
        image (tf.Tensor): Input image tensor (H, W, C)
        mask (tf.Tensor): Segmentation mask (H, W)
        crop_size (int): Crop size (256 or 384)

    Returns:
        Tuple:
            image_crop (tf.Tensor): Cropped and resized image (512,512,C)
            mask_crop  (tf.Tensor): Cropped and resized mask (512,512)

    Cropping Priority:
    ------------------
    1. Destroyed pixels (mask == 3)
    2. Any damage pixels (mask > 0)
    3. Random crop (fallback if no damage exists)

    Pipeline:
    ---------
    1. Identify candidate coordinates:
        - destroyed_coords
        - damage_coords

    2. Select coordinate set:
        - Prefer destroyed if available
        - Otherwise use general damage

    3. Crop centered around selected pixel

    4. Fallback:
        - If no damage → random crop over full image

    5. Resize:
        - All crops resized back to 512×512 (critical)

    Notes:
        - Ensures consistent input size for model
        - Uses nearest-neighbor interpolation for masks
        - Improves sampling of rare classes
    """

    mask = tf.cast(mask, tf.int32)

    destroyed_coords = tf.where(mask == 3)
    damage_coords = tf.where(mask > 0)

    # -------------------------------------------------
    # SELECT WHICH COORDS TO USE
    # -------------------------------------------------
    coords = tf.cond(
        tf.shape(destroyed_coords)[0] > 0,
        lambda: destroyed_coords,
        lambda: damage_coords
    )

    def crop_on_damage():
        idx = tf.random.uniform(
            [], 0, tf.shape(coords)[0], dtype=tf.int32
        )
        center = coords[idx]

        y, x = center[0], center[1]

        y1 = tf.clip_by_value(y - crop_size // 2, 0, IMAGE_SIZE - crop_size)
        x1 = tf.clip_by_value(x - crop_size // 2, 0, IMAGE_SIZE - crop_size)

        image_crop = image[y1:y1 + crop_size, x1:x1 + crop_size, :]
        mask_crop = mask[y1:y1 + crop_size, x1:x1 + crop_size]

        return image_crop, mask_crop

    def random_crop():
        """
        Random fallback crop (no damage at all)
        """

        mask_float = tf.cast(mask, tf.float32)

        combined = tf.concat(
            [image, tf.expand_dims(mask_float, -1)],
            axis=-1
        )

        cropped = tf.image.random_crop(
            combined,
            size=[crop_size, crop_size, tf.shape(combined)[-1]]
        )

        image_crop = cropped[..., :-1]
        mask_crop = tf.cast(cropped[..., -1], tf.int32)

        return image_crop, mask_crop

    image_crop, mask_crop = tf.cond(
        tf.shape(coords)[0] > 0,
        crop_on_damage,
        random_crop
    )

    # -------------------------------------------------
    # Resize back to 512 (CRITICAL)
    # -------------------------------------------------
    image_crop = tf.image.resize(image_crop, (IMAGE_SIZE, IMAGE_SIZE))

    mask_crop = tf.image.resize(
        tf.expand_dims(mask_crop, -1),
        (IMAGE_SIZE, IMAGE_SIZE),
        method="nearest"
    )
    mask_crop = tf.squeeze(mask_crop, axis=-1)

    mask_crop = tf.cast(mask_crop, tf.int32)

    return image_crop, mask_crop


# =========================================================
# ADAPTIVE MULTISCALE SAMPLING
# =========================================================
def _apply_multiscale_adaptive(image, mask):
    """
    Apply adaptive multiscale sampling to input data.

    Args:
        image (tf.Tensor): Input image (512,512,C)
        mask (tf.Tensor): Segmentation mask (512,512)

    Returns:
        Tuple:
            image (tf.Tensor): Transformed image
            mask  (tf.Tensor): Transformed mask

    Strategy:
    ---------
    Randomly selects one of three scales:

    - 50% → full image (no cropping)
    - 30% → medium crop (384 → resized to 512)
    - 20% → small crop (256 → resized to 512)

    Benefits:
    ---------
    - Introduces scale diversity
    - Encourages robustness to object size variations
    - Enhances learning of small damage regions

    Notes:
        - Crops use damage-aware sampling
        - All outputs maintain consistent shape (512×512)
    """

    mask = tf.cast(mask, tf.int32)
    rand = tf.random.uniform([])

    def full():
        return image, mask

    def medium():
        return _damage_aware_crop(image, mask, CROP_MEDIUM)

    def small():
        return _damage_aware_crop(image, mask, CROP_SMALL)

    return tf.cond(
        rand < 0.5,
        full,
        lambda: tf.cond(
            rand < 0.8,
            medium,
            small
        )
    )


# =========================================================
# DATASET BUILDER
# =========================================================
def get_dataset(
    tfrecord_dir,
    batch_size=8,
    shuffle=True
):
    """
    Build TF dataset with multiscale adaptive damage-aware sampling.

    Args:
        tfrecord_dir (str):
            Directory containing TFRecord shards
        batch_size (int):
            Number of samples per batch (default: 8)
        shuffle (bool):
            Whether to shuffle dataset (default: True)

    Returns:
        tf.data.Dataset:
            Dataset yielding (image, mask) pairs

    Pipeline:
    ---------
    1. Load TFRecord files
    2. Parse examples using _parse_example
    3. Enforce mask dtype (int32)
    4. Optional shuffle
    5. Apply multiscale adaptive sampling
    6. Batch dataset
    7. Prefetch for performance

    Output:
    -------
    image: (512,512,22) float32  
    mask : (512,512) int32  

    Performance Features:
    ---------------------
    - Parallel mapping (AUTOTUNE)
    - Prefetching for GPU utilization
    - Lightweight augmentation via cropping

    Notes:
        - Designed for training only (not evaluation)
        - Maintains deterministic shape despite cropping
        - Integrates seamlessly with TFRecord pipeline

    Run Example:
    ------------
    train_ds = get_dataset(
        tfrecord_dir="tfrecords/train/balanced",
        batch_size=8,
        shuffle=True
    )
    """

    files = tf.io.gfile.glob(f"{tfrecord_dir}/*.tfrecord")

    dataset = tf.data.TFRecordDataset(files)

    dataset = dataset.map(
        _parse_example,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # enforce dtype early
    dataset = dataset.map(
        lambda x, y: (x, tf.cast(y, tf.int32)),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=256)

    dataset = dataset.map(
        _apply_multiscale_adaptive,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset