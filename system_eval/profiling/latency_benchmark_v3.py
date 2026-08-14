#!/usr/bin/env python3
"""
GPU Latency Benchmark — v3 (FINAL)

Overview:
---------
This script benchmarks inference performance of the final segmentation model
(Dual-Branch Efficient GATED DIFF v3) on GPU.

It measures:
    - Single-image latency (batch size = 1)
    - Batch latency (batch size = 4)
    - Throughput (images per second)

The benchmark is aligned with v2 to ensure fair comparison across model versions.

Purpose:
--------
Evaluate production readiness of the model by quantifying:
    - Inference speed
    - Scalability with batching
    - Stability of latency (std deviation)

Metrics:
--------
- latency_batch1_ms:
    Average latency for batch size 1 (milliseconds)

- latency_batch4_ms:
    Average latency for batch size 4 (milliseconds)

- throughput_img_per_sec:
    Number of images processed per second

- std_batch1_ms / std_batch4_ms:
    Standard deviation of latency (stability measure)

Benchmark Methodology:
----------------------
1. Warmup runs (GPU graph compilation + cache stabilization)
2. Timed inference runs (MEASURE_RUNS)
3. GPU synchronization after each run for accurate timing
4. Mean and std computed over multiple runs

Run Example:
------------
PYTHONPATH=. python system_eval_v2/profiling/latency_benchmark_v3.py

Output:
-------
Results saved to:
    system_eval_v2/profiling/results/latency_gpu_v3.json

Notes:
------
- Uses tf.function for graph execution (realistic inference)
- GPU memory growth enabled to avoid allocation spikes
- Random input used to simulate real inference workload
"""

import tensorflow as tf
import numpy as np
import time
import json
import os

#  IMPORTANT: ensure model is registered
import src.models.dual_branch_diff_gated_segmentation_model_v3

# =========================================================
# CONFIG
# =========================================================
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
    Number of warmup iterations before timing
    Ensures:
        - graph compilation
        - kernel caching
        - stable GPU execution

MEASURE_RUNS:
    Number of timed inference runs
    Higher value → more stable statistics

OUTPUT_PATH:
    File path to save benchmark results (JSON)

Notes:
------
- Values chosen to balance accuracy vs runtime
- Warmup is critical for fair benchmarking
"""
MODEL_PATH = "experiments_v2/dual_branch_diff_gated_v3_20260406-224335/best_model.keras"

WARMUP_RUNS = 10
MEASURE_RUNS = 50

OUTPUT_PATH = "system_eval_v2/profiling/results/latency_gpu_v3.json"


# =========================================================
# GPU SETUP
# =========================================================
"""
GPU configuration.

Behavior:
---------
- Detects available GPUs
- Enables memory growth to prevent:
    - full pre-allocation
    - out-of-memory spikes

Benefits:
---------
- More stable benchmarking
- Better compatibility across environments

Notes:
------
- Required when running multiple experiments
- Prevents TensorFlow from reserving all GPU memory upfront
"""
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# =========================================================
# BENCHMARK FUNCTION
# =========================================================
def benchmark(model, input_data):
    """
    Measure inference latency for a given model and input.

    Args:
        model (tf.keras.Model):
            Loaded segmentation model
        input_data (np.ndarray):
            Input tensor of shape (B, 512, 512, 22)

    Returns:
        Tuple:
            mean_time (float): Average inference time (seconds)
            std_time  (float): Standard deviation (seconds)

    Pipeline:
    ---------
    1. Wrap inference in tf.function:
        → Enables graph execution (faster, realistic)

    2. Warmup phase:
        → Run model multiple times without timing
        → Stabilizes GPU kernels and execution graph

    3. Synchronization:
        → Ensures all GPU operations complete before timing

    4. Measurement phase:
        → Run inference multiple times
        → Record execution time per run

    5. Compute statistics:
        → mean latency
        → standard deviation

    Notes:
        - tf.experimental.sync_devices() ensures accurate GPU timing
        - Without synchronization, timing would be incorrect (async execution)
        - Batch size impacts both latency and throughput
    """

    @tf.function
    def infer(x):
        return model(x, training=False)

    # Warmup
    for _ in range(WARMUP_RUNS):
        _ = infer(input_data)

    #  Ensure warmup ops finish
    tf.experimental.sync_devices() if hasattr(tf.experimental, "sync_devices") else None

    # Measure
    times = []
    for _ in range(MEASURE_RUNS):
        start = time.time()
        _ = infer(input_data)
        tf.experimental.sync_devices() if hasattr(tf.experimental, "sync_devices") else None
        times.append(time.time() - start)

    return np.mean(times), np.std(times)


# =========================================================
# MAIN
# =========================================================
def main():
    """
    Main benchmarking pipeline.

    Workflow:
    ---------
    1. Load trained model (v3)
    2. Generate synthetic input tensors
    3. Run latency benchmark for:
        - batch size = 1
        - batch size = 4
    4. Compute:
        - latency (ms)
        - throughput (images/sec)
    5. Save results to JSON
    6. Print results

    Steps:
    ------

    [1] Model Loading
        - Loads .keras model
        - compile=False for faster loading

    [2] Input Preparation
        - Random tensors simulate real inputs
        - Shapes:
            (1, 512, 512, 22)
            (4, 512, 512, 22)

    [3] Benchmark Execution
        - Calls benchmark() for each batch size

    [4] Metric Computation
        - Convert latency to milliseconds
        - Compute throughput:
            throughput = batch_size / latency

    [5] Output Saving
        - Saves results as JSON file
        - Ensures output directory exists

    Notes:
    ------
    - Designed for GPU benchmarking
    - Synthetic inputs remove I/O bottlenecks
    - Enables fair comparison across model versions
    """

    print("\n Loading v3 model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False
    )

    print(" Model loaded.\n")

    # Inputs
    input_1 = np.random.rand(1, 512, 512, 22).astype(np.float32)
    input_4 = np.random.rand(4, 512, 512, 22).astype(np.float32)

    print(" Running benchmark...\n")

    t1_mean, t1_std = benchmark(model, input_1)
    t4_mean, t4_std = benchmark(model, input_4)

    latency_ms = t1_mean * 1000
    latency4_ms = t4_mean * 1000
    throughput = 4 / t4_mean

    results = {
        "model": "dual_branch_diff_gated_v3",
        "latency_batch1_ms": round(latency_ms, 2),
        "latency_batch4_ms": round(latency4_ms, 2),
        "throughput_img_per_sec": round(throughput, 2),
        "std_batch1_ms": round(t1_std * 1000, 2),
        "std_batch4_ms": round(t4_std * 1000, 2)
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print(" LATENCY RESULTS (v3)")
    print(json.dumps(results, indent=4))


# =========================================================
# ENTRY
# =========================================================
if __name__ == "__main__":
    """
    Entry point for latency benchmark script.

    Executes:
        main()

    Usage:
    ------
    PYTHONPATH=. python system_eval_v2/profiling/latency_benchmark_v3.py

    Notes:
        - Ensure GPU is available
        - Model path must be valid
        - Results will be saved automatically
    """
    main()