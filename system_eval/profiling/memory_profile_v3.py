#!/usr/bin/env python3
"""
Memory Profiling — v3 (GPU, FINAL)

Overview:
---------
This script measures GPU memory usage of the final segmentation model
(Dual-Branch Efficient GATED DIFF v3) during inference.

It reports:
    - Peak GPU memory usage (MB)
    - Current GPU memory usage (MB)

Purpose:
--------
Evaluate model deployment feasibility by quantifying:
    - Memory footprint during inference
    - Scalability constraints on GPU hardware
    - Suitability for production environments

Metrics:
--------
- peak_gpu_memory_mb:
    Maximum memory allocated during execution

- current_gpu_memory_mb:
    Memory allocated after inference completes

Improvements over v2:
---------------------
- Explicit reset of GPU memory stats before measurement
- Warmup runs to stabilize memory allocation
- More reliable and reproducible peak measurement

Benchmark Methodology:
----------------------
1. Initialize GPU with memory growth enabled
2. Load trained model
3. Reset GPU memory statistics
4. Perform warmup runs
5. Run inference
6. Query GPU memory usage via TensorFlow API
7. Compute and report results

Run Example:
------------
PYTHONPATH=. python system_eval_v2/profiling/memory_profile_v3.py

Output:
-------
Results saved to:
    system_eval_v2/profiling/results/memory_v3.json

Notes:
------
- Requires GPU-enabled TensorFlow
- Uses synthetic input to isolate model memory usage
- Measures inference-time memory (not training)
"""

import tensorflow as tf
import numpy as np
import json
import os

#  IMPORTANT: register model
import src.models.dual_branch_diff_gated_segmentation_model_v3

# =========================================================
# CONFIG
# =========================================================
"""
Memory profiling configuration.

Fields:
-------
MODEL_PATH:
    Path to trained model (.keras)

OUTPUT_PATH:
    Path to save memory profiling results (JSON)

WARMUP_RUNS:
    Number of warmup iterations before measurement

Purpose:
--------
- Warmup ensures stable memory allocation
- Resetting stats ensures clean measurement window

Notes:
------
- Warmup is critical to capture realistic peak usage
"""
MODEL_PATH = "experiments_v2/dual_branch_diff_gated_v3_20260406-224335/best_model.keras"
OUTPUT_PATH = "system_eval_v2/profiling/results/memory_v3.json"

WARMUP_RUNS = 5


# =========================================================
# MAIN
# =========================================================
def main():
    """
    Main GPU memory profiling pipeline.

    Workflow:
    ---------
    1. Detect and configure GPU
    2. Load trained model
    3. Prepare synthetic input tensor
    4. Reset GPU memory statistics
    5. Perform warmup runs
    6. Run inference for measurement
    7. Retrieve memory usage statistics
    8. Save results to JSON
    9. Print results

    Steps:
    ------

    [1] GPU Setup
        - Detect available GPU devices
        - Enable memory growth to prevent full allocation

    [2] Model Loading
        - Load trained model from disk
        - compile=False for faster initialization

    [3] Input Preparation
        - Generate random input tensor
        - Shape: (1, 512, 512, 22)

    [4] Reset Memory Stats (CRITICAL)
        - Clears previous memory usage records
        - Ensures accurate measurement window

    [5] Warmup Phase
        - Run inference multiple times
        - Stabilizes memory allocation and kernel execution

    [6] Measurement Phase
        - Run inference once
        - Capture memory statistics immediately after

    [7] Memory Retrieval
        - Use TensorFlow API:
            tf.config.experimental.get_memory_info('GPU:0')

        Returns:
            - 'peak'    → max memory usage
            - 'current' → current memory usage

    [8] Conversion
        - Convert bytes → MB for readability

    [9] Output Saving
        - Save results as JSON file
        - Ensure output directory exists

    Notes:
    ------
    - Peak memory reflects worst-case allocation during inference
    - Current memory reflects steady-state allocation
    - Important for deployment constraints and model scaling
    """

    print("\n Setting up GPU...")

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        print(" No GPU found")
        return

    tf.config.experimental.set_memory_growth(gpus[0], True)

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------
    print(" Loading v3 model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False
    )

    print(" Model loaded.\n")

    # -----------------------------------------------------
    # Input
    # -----------------------------------------------------
    input_data = np.random.rand(1, 512, 512, 22).astype(np.float32)

    # -----------------------------------------------------
    # Reset memory stats (CRITICAL)
    # -----------------------------------------------------
    tf.config.experimental.reset_memory_stats('GPU:0')

    # -----------------------------------------------------
    # Warmup
    # -----------------------------------------------------
    print(" Running warmup...")

    for _ in range(WARMUP_RUNS):
        _ = model(input_data, training=False)

    # -----------------------------------------------------
    # Actual measurement
    # -----------------------------------------------------
    print(" Measuring memory...")

    _ = model(input_data, training=False)

    mem_info = tf.config.experimental.get_memory_info('GPU:0')

    peak_mb = mem_info['peak'] / (1024 ** 2)
    current_mb = mem_info['current'] / (1024 ** 2)

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------
    results = {
        "model": "dual_branch_diff_gated_v3",
        "peak_gpu_memory_mb": round(peak_mb, 2),
        "current_gpu_memory_mb": round(current_mb, 2)
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print("\n MEMORY PROFILE (v3)")
    print(json.dumps(results, indent=4))


# =========================================================
# ENTRY
# =========================================================
if __name__ == "__main__":
    """
    Entry point for memory profiling script.

    Executes:
        main()

    Usage:
    ------
    PYTHONPATH=. python system_eval_v2/profiling/memory_profile_v3.py

    Notes:
        - Requires GPU availability
        - Model path must be valid
        - Results saved automatically
    """
    main()