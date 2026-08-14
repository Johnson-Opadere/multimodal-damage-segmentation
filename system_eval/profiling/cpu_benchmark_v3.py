#!/usr/bin/env python3
"""
CPU Latency Benchmark — v3 (FINAL)

Overview:
---------
This script benchmarks inference latency of the final segmentation model
(Dual-Branch Efficient GATED DIFF v3) on CPU.

It forces CPU-only execution and measures:
    - Single-image inference latency (ms)
    - Latency stability (standard deviation)

The benchmark is aligned with v2 for fair comparison across model versions.

Purpose:
--------
Evaluate deployment feasibility in CPU-only environments such as:
    - Edge devices
    - Low-resource servers
    - Cost-sensitive production systems

Metrics:
--------
- cpu_latency_ms:
    Average inference time per image (milliseconds)

- std_ms:
    Standard deviation of latency (stability measure)

Benchmark Methodology:
----------------------
1. Force CPU-only execution (disable GPU)
2. Warmup runs to stabilize execution
3. Timed inference runs
4. Compute mean and standard deviation

Run Example:
------------
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python system_eval_v2/profiling/cpu_benchmark_v3.py

Output:
-------
Results saved to:
    system_eval_v2/profiling/results/latency_cpu_v3.json

Notes:
------
- Uses synthetic input to isolate model inference cost
- No tf.function used (reflects eager CPU execution)
- Designed for consistent comparison with GPU benchmarks
"""

# =========================
#  FORCE CPU (CRITICAL)
# =========================
"""
Force TensorFlow to run on CPU only.

Mechanism:
----------
Sets environment variable:
    CUDA_VISIBLE_DEVICES = ""

Effect:
-------
- Disables all GPU devices
- Ensures inference runs strictly on CPU

Why this matters:
-----------------
- Prevents accidental GPU usage
- Ensures accurate CPU latency measurement
- Enables fair comparison across hardware setups

Notes:
------
- Must be set BEFORE TensorFlow is imported
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
import numpy as np
import time
import json

#  Ensure model loads correctly
import src.models.dual_branch_diff_gated_segmentation_model_v3

# =========================================================
# CONFIG
# =========================================================
"""
Benchmark configuration.

Fields:
-------
MODEL_PATH:
    Path to trained model (.keras)

WARMUP_RUNS:
    Number of warmup iterations
    Allows CPU execution to stabilize

MEASURE_RUNS:
    Number of timed inference runs
    More runs → more reliable statistics

OUTPUT_PATH:
    JSON file path for saving results

Notes:
------
- CPU benchmarking requires fewer runs than GPU
- Values chosen to balance speed vs accuracy
"""
MODEL_PATH = "experiments_v2/dual_branch_diff_gated_v3_20260406-224335/best_model.keras"

WARMUP_RUNS = 5
MEASURE_RUNS = 20

OUTPUT_PATH = "system_eval_v2/profiling/results/latency_cpu_v3.json"


# =========================================================
# MAIN
# =========================================================
def main():
    """
    Main CPU benchmarking pipeline.

    Workflow:
    ---------
    1. Load trained model in CPU mode
    2. Generate synthetic input tensor
    3. Perform warmup runs
    4. Measure inference latency
    5. Compute statistics
    6. Save results to JSON
    7. Print results

    Steps:
    ------

    [1] Model Loading
        - Loads trained model from disk
        - compile=False for faster loading

    [2] Input Preparation
        - Random tensor simulating real input
        - Shape: (1, 512, 512, 22)

    [3] Warmup Phase
        - Runs model multiple times without timing
        - Stabilizes execution (CPU caching, memory allocation)

    [4] Measurement Phase
        - Records inference time per run
        - Uses time.time() for timing

    [5] Metric Computation
        - mean_time → average latency
        - std_time → variability

    [6] Output Saving
        - Saves results as JSON
        - Ensures output directory exists

    Notes:
    ------
    - No GPU synchronization needed (CPU execution is synchronous)
    - Synthetic inputs remove I/O overhead
    - Suitable for benchmarking deployment scenarios
    """

    print("\n Loading v3 model (CPU mode)...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False
    )

    print(" Model loaded.\n")

    input_data = np.random.rand(1, 512, 512, 22).astype(np.float32)

    # -----------------------------------------------------
    # Warmup
    # -----------------------------------------------------
    print(" Running warmup...\n")

    for _ in range(WARMUP_RUNS):
        _ = model(input_data, training=False)

    # -----------------------------------------------------
    # Measure
    # -----------------------------------------------------
    print(" Measuring CPU latency...\n")

    times = []

    for _ in range(MEASURE_RUNS):
        start = time.time()
        _ = model(input_data, training=False)
        times.append(time.time() - start)

    mean_time = np.mean(times)
    std_time = np.std(times)

    latency_ms = mean_time * 1000

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------
    results = {
        "model": "dual_branch_diff_gated_v3",
        "cpu_latency_ms": round(latency_ms, 2),
        "std_ms": round(std_time * 1000, 2)
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print(" CPU LATENCY RESULTS (v3)")
    print(json.dumps(results, indent=4))


# =========================================================
# ENTRY
# =========================================================
if __name__ == "__main__":
    """
    Entry point for CPU latency benchmark.

    Executes:
        main()

    Usage:
    ------
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python system_eval_v2/profiling/cpu_benchmark_v3.py

    Notes:
        - Ensures CPU-only execution
        - Model path must be valid
        - Results saved automatically
    """
    main()