#!/usr/bin/env python3
"""
Model Size Profiling — v3 (FINAL - ROBUST)

Overview:
---------
This script profiles the size and parameter footprint of the final segmentation
model (Dual-Branch Efficient GATED DIFF v3).

It measures:
    - Total number of parameters
    - Number of trainable parameters
    - Serialized model size on disk (MB)

Purpose:
--------
Evaluate model efficiency and deployment feasibility by quantifying:
    - Memory footprint (storage)
    - Model complexity (parameter count)
    - Training vs inference parameter distribution

Metrics:
--------
- total_params:
    Total number of parameters in the model

- trainable_params:
    Number of parameters updated during training

- model_size_mb:
    Size of saved model file on disk (MB)

Fixes / Improvements over v2:
-----------------------------
- Forces CPU execution to avoid GPU/cuDNN stalls
- Handles custom layers safely via module import
- Ensures consistent model loading across environments

Run Example:
------------
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python system_eval_v2/profiling/model_size_v3.py

Output:
-------
Results saved to:
    system_eval_v2/profiling/results/model_size_v3.json

Notes:
------
- CPU mode ensures stable model loading
- Parameter count reflects model complexity
- File size reflects deployment/storage cost
"""

# =========================
#  FORCE CPU (CRITICAL)
# =========================
"""
Force TensorFlow to run in CPU-only mode.

Mechanism:
----------
Sets environment variable:
    CUDA_VISIBLE_DEVICES = ""

Effect:
-------
- Disables all GPU devices
- Prevents GPU-related initialization issues (e.g., cuDNN stalls)

Why this matters:
-----------------
- Ensures stable model loading
- Avoids GPU dependency for profiling
- Makes script portable across environments

Notes:
------
- Must be set BEFORE TensorFlow import
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# =========================
# Imports
# =========================
import tensorflow as tf
import json

# =========================
#  IMPORT MODEL MODULE (CRITICAL)
# =========================
"""
Import model module to register custom layers/functions.

Purpose:
--------
Ensures that all model components are properly registered before loading.

Why needed:
-----------
- Keras requires access to custom layers/functions at load time
- Prevents deserialization errors

Notes:
------
- Even if no custom layers exist, this is kept for robustness
"""
# Ensures all layers / functions are registered
import src.models.dual_branch_diff_gated_segmentation_model_v3 as model_module

# =========================
# CONFIG
# =========================
"""
Import model module to register custom layers/functions.

Purpose:
--------
Ensures that all model components are properly registered before loading.

Why needed:
-----------
- Keras requires access to custom layers/functions at load time
- Prevents deserialization errors

Notes:
------
- Even if no custom layers exist, this is kept for robustness
"""
MODEL_PATH = "experiments_v2/dual_branch_diff_gated_v3_20260406-224335/best_model.keras"

OUTPUT_PATH = "system_eval_v2/profiling/results/model_size_v3.json"


# =========================
# UTIL
# =========================
def get_model_size_mb(path):
    """
    Compute model file size in megabytes.

    Args:
        path (str): Path to model file

    Returns:
        float: File size in MB

    Method:
    -------
    - Retrieve file size in bytes
    - Convert to MB (1 MB = 1024^2 bytes)

    Notes:
        - Reflects storage cost of model
        - Useful for deployment constraints
    """
    size_bytes = os.path.getsize(path)
    return size_bytes / (1024 * 1024)


# =========================
# MAIN
# =========================
def main():
    """
    Main model size profiling pipeline.

    Workflow:
    ---------
    1. Load trained model in CPU mode
    2. Compute parameter counts
    3. Compute model file size
    4. Save results to JSON
    5. Print results

    Steps:
    ------

    [1] Model Loading
        - Load model using tf.keras.models.load_model
        - compile=False for faster loading
        - safe_mode=False for compatibility
        - custom_objects provided for robustness

    [2] Parameter Counting
        - total_params:
            model.count_params()
        - trainable_params:
            Sum of all trainable variables

    [3] File Size Computation
        - Compute size of serialized model file

    [4] Output Saving
        - Save results as JSON
        - Ensure output directory exists

    Notes:
    ------
    - Parameter count reflects model complexity
    - Trainable params reflect optimization scope
    - File size reflects deployment/storage cost
    """
    print("\n Loading v3 model (CPU mode)...")

    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            safe_mode=False,   # safe + consistent with v2
            custom_objects={}  # v3 has no Lambda, but kept for safety
        )
    except Exception as e:
        print("\n Model loading failed.")
        print("Error:", e)
        return

    print(" Model loaded successfully.\n")

    # =========================
    # PARAM COUNT
    # =========================
    print(" Computing parameter counts...")

    total_params = model.count_params()

    trainable_params = int(
        sum([tf.size(v).numpy() for v in model.trainable_variables])
    )

    # =========================
    # FILE SIZE
    # =========================
    print(" Computing model file size...")

    size_mb = get_model_size_mb(MODEL_PATH)

    # =========================
    # RESULTS
    # =========================
    results = {
        "model": "dual_branch_diff_gated_v3",
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "model_size_mb": round(size_mb, 2)
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print("\n MODEL SIZE RESULTS (v3)")
    print(json.dumps(results, indent=4))


# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    """
    Entry point for model size profiling script.

    Executes:
        main()

    Usage:
    ------
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python system_eval_v2/profiling/model_size_v3.py

    Notes:
        - Runs entirely on CPU
        - Model path must be valid
        - Results saved automatically
    """
    main()