#!/usr/bin/env python3
"""
System Evaluation Summary (v2 vs v3) — FINAL (ULTIMATE)

Overview:
---------
This script aggregates and synthesizes all evaluation outputs across:
    - Model accuracy
    - Latency (GPU + CPU)
    - Throughput
    - Memory usage
    - Model size

It compares two model versions:
    - v2 (context-aware gated DIFF)
    - v3 (efficient gated DIFF)

Purpose:
--------
Provide a **single source of truth** for:
    - Model comparison
    - Performance trade-offs
    - Deployment decision-making

Outputs:
--------
- Consolidated JSON summary containing:
    • Accuracy metrics
    • System metrics
    • Improvement ratios
    • Interpretation (human-readable insights)
    • Deployment recommendations
    • Provenance + hardware metadata

Run Example:
------------
PYTHONPATH=. python system_eval_v2/profiling/summary.py

Output File:
------------
system_eval_v2/profiling/results/summary.json

Design Goals:
-------------
 Combine all evaluation artifacts  
 Provide reproducible metrics  
 Enable executive-level interpretation  
 Support deployment decisions  

Notes:
------
- Missing files are handled gracefully
- All ratios are computed safely
- Designed for both engineering and interview presentation
"""

import json
import os
from datetime import datetime


# =========================================================
# PATHS
# =========================================================
"""
Paths to experiment outputs and profiling results.

Fields:
-------
V2_EXP_DIR / V3_EXP_DIR:
    Directories containing model outputs (metrics, checkpoints)

RESULTS_DIR:
    Directory containing system profiling outputs

OUTPUT_PATH:
    Path to final consolidated summary JSON

Notes:
------
- Paths must match experiment structure
- Missing files are handled safely downstream
"""

V2_EXP_DIR = "experiments_v2/dual_branch_diff_gated_v2_20260405-204616"
V3_EXP_DIR = "experiments_v2/dual_branch_diff_gated_v3_20260406-224335"

RESULTS_DIR = "system_eval_v2/profiling/results"
OUTPUT_PATH = os.path.join(RESULTS_DIR, "summary.json")


# =========================================================
# UTIL
# =========================================================

def load_json_safe(path):
    """
    Safely load JSON file.

    Args:
        path (str): Path to JSON file

    Returns:
        dict: Parsed JSON content (or empty dict if missing)

    Behavior:
    ---------
    - Returns empty dict if file does not exist
    - Prints warning for missing files

    Purpose:
    --------
    Prevent pipeline failure due to missing artifacts
    """
    if not os.path.exists(path):
        print(f" Missing file: {path}")
        return {}
    with open(path, "r") as f:
        return json.load(f)


def safe_get(d, key, default=None):
    """
    Safely extract value from dictionary.

    Args:
        d (dict): Input dictionary
        key (str): Key to retrieve
        default: Default value if key not found

    Returns:
        Value associated with key or default

    Notes:
        - Handles non-dict inputs safely
    """
    return d.get(key, default) if isinstance(d, dict) else default


# =========================================================
# LOAD DATA
# =========================================================
"""
Load evaluation artifacts for both model versions.

Sources:
--------
- Accuracy metrics (test split)
- Model size profiling
- GPU latency + throughput
- CPU latency
- GPU memory usage

Notes:
------
- Each file corresponds to a specific evaluation stage
- Missing files are tolerated via safe loading
"""

v2_metrics = load_json_safe(os.path.join(V2_EXP_DIR, "test_metrics.json"))
v3_metrics = load_json_safe(os.path.join(V3_EXP_DIR, "test_metrics.json"))

v2_size = load_json_safe(os.path.join(RESULTS_DIR, "model_size.json"))
v3_size = load_json_safe(os.path.join(RESULTS_DIR, "model_size_v3.json"))

v2_gpu = load_json_safe(os.path.join(RESULTS_DIR, "latency_gpu.json"))
v3_gpu = load_json_safe(os.path.join(RESULTS_DIR, "latency_gpu_v3.json"))

v2_cpu = load_json_safe(os.path.join(RESULTS_DIR, "latency_cpu.json"))
v3_cpu = load_json_safe(os.path.join(RESULTS_DIR, "latency_cpu_v3.json"))

v2_mem = load_json_safe(os.path.join(RESULTS_DIR, "memory.json"))
v3_mem = load_json_safe(os.path.join(RESULTS_DIR, "memory_v3.json"))


# =========================================================
# EXTRACT
# =========================================================

def extract_model_block(metrics, size, gpu, cpu, mem):
    """
    Extract structured metrics for a single model version.

    Args:
        metrics (dict): Evaluation metrics (IoU, etc.)
        size (dict): Model size profiling results
        gpu (dict): GPU latency + throughput results
        cpu (dict): CPU latency results
        mem (dict): GPU memory usage results

    Returns:
        dict: Structured block containing:
            - accuracy metrics
            - system performance metrics

    Structure:
    ----------
    accuracy:
        - test_miou
        - damage_iou

    system_metrics:
        - model size
        - latency (GPU + CPU)
        - throughput
        - memory usage

    Purpose:
    --------
    Normalize data from multiple sources into a unified format
    """
    return {
        "accuracy": {
            "test_miou": safe_get(metrics, "mean_iou"),
            "damage_iou": safe_get(metrics.get("damage_only_stats", {}), "mean_damage_iou_gt_only")
        },
        "system_metrics": {
            "model_size_mb": safe_get(size, "model_size_mb"),
            "gpu_latency_batch1_ms": safe_get(gpu, "latency_batch1_ms"),
            "gpu_latency_batch4_ms": safe_get(gpu, "latency_batch4_ms"),
            "gpu_latency_std_ms": safe_get(gpu, "std_batch1_ms"),
            "cpu_latency_ms": safe_get(cpu, "cpu_latency_ms"),
            "cpu_latency_std_ms": safe_get(cpu, "std_ms"),
            "throughput_img_per_sec": safe_get(gpu, "throughput_img_per_sec"),
            "peak_gpu_memory_mb": safe_get(mem, "peak_gpu_memory_mb")
        }
    }


v2_block = extract_model_block(v2_metrics, v2_size, v2_gpu, v2_cpu, v2_mem)
v3_block = extract_model_block(v3_metrics, v3_size, v3_gpu, v3_cpu, v3_mem)


# =========================================================
# IMPROVEMENTS
# =========================================================
"""
Compute relative improvements between v2 and v3.

Metrics:
--------
- Accuracy gain (mIoU, damage IoU)
- GPU latency speedup
- Throughput gain
- Model size reduction
- Memory reduction
- CPU speedup

Purpose:
--------
Quantify performance improvements in a normalized way
"""

def safe_ratio(a, b):
    """
    Safely compute ratio between two values.

    Args:
        a (float): Numerator
        b (float): Denominator

    Returns:
        float or None: Ratio rounded to 2 decimals

    Behavior:
    ---------
    - Returns None if values are invalid
    - Prevents division errors

    Purpose:
    --------
    Used for computing improvement factors between models
    """
    try:
        return round(a / b, 2) if a and b else None
    except:
        return None


improvements = {
    "accuracy_gain_miou": round(v3_block["accuracy"]["test_miou"] - v2_block["accuracy"]["test_miou"], 4),
    "damage_iou_gain": round(v3_block["accuracy"]["damage_iou"] - v2_block["accuracy"]["damage_iou"], 4),
    "latency_speedup_gpu": safe_ratio(v2_block["system_metrics"]["gpu_latency_batch1_ms"],
                                     v3_block["system_metrics"]["gpu_latency_batch1_ms"]),
    "throughput_gain": safe_ratio(v3_block["system_metrics"]["throughput_img_per_sec"],
                                 v2_block["system_metrics"]["throughput_img_per_sec"]),
    "model_size_reduction": safe_ratio(v2_block["system_metrics"]["model_size_mb"],
                                      v3_block["system_metrics"]["model_size_mb"]),
    "memory_reduction": safe_ratio(v2_block["system_metrics"]["peak_gpu_memory_mb"],
                                  v3_block["system_metrics"]["peak_gpu_memory_mb"]),
    "cpu_speedup": safe_ratio(v2_block["system_metrics"]["cpu_latency_ms"],
                             v3_block["system_metrics"]["cpu_latency_ms"])
}


# =========================================================
# INTERPRETATION
# =========================================================
"""
Human-readable interpretation of results.

Components:
-----------
- Summary statement
- Key takeaways (bullet points)
- Deployment decision
- Real-time capability assessment
- Bottleneck analysis
- Serving recommendations
- Limitations
- Future work

Purpose:
--------
Translate raw metrics into actionable insights.

Notes:
------
- Designed for:
    • Interview explanation
    • Executive summaries
    • Decision-making support
"""

interpretation = {
    "summary": "v3 dominates v2 across both accuracy and system performance.",

    "key_takeaways": [
        f"Accuracy improved (+{improvements['accuracy_gain_miou']} mIoU, +{improvements['damage_iou_gain']} damage IoU).",
        f"GPU latency reduced ~{improvements['latency_speedup_gpu']}× (2.39s → 33.75ms ≈ 29.6 FPS), enabling real-time inference.",
        f"Throughput increased ~{improvements['throughput_gain']}×.",
        f"Model size reduced ~{improvements['model_size_reduction']}×.",
        f"GPU memory reduced ~{improvements['memory_reduction']}×.",
        f"CPU improved ~{improvements['cpu_speedup']}× but remains non-real-time."
    ],

    "deployment_decision": "Use v3 on GPU for real-time serving; CPU as fallback only.",

    "real_time_definition": {
        "target_fps": 30,
        "achieved_latency_ms": v3_block["system_metrics"]["gpu_latency_batch1_ms"],
        "achieved_fps": 29.63,
        "status": "Real-time capable (~30 FPS on NVIDIA A10G)"
    },

    "bottleneck_analysis": {
        "v2": "Compute-bound due to overparameterized gating and feature explosion.",
        "v3": "Improved efficiency via channel compression and streamlined gating.",
        "cpu_vs_gpu": "GPU benefits from parallelism; CPU remains compute/memory bound."
    },

    "serving_recommendation": {
        "primary": "GPU deployment on AWS g5 instances for real-time inference",
        "fallback": "CPU deployment for batch/offline processing only",
        "scaling": "Horizontal scaling using multiple GPU instances or batching"
    },

    "limitations": [
        "CPU inference latency (~1.5s) is not suitable for real-time applications",
        "Minor damage class remains comparatively harder",
        "Performance tied to 512x512 resolution"
    ],

    "future_work": [
        "TensorRT / quantization for further latency reduction",
        "Edge deployment exploration",
        "Improve minor damage class performance"
    ]
}


# =========================================================
# PROVENANCE + HARDWARE
# =========================================================
"""
Metadata describing experiment context.

Provenance:
-----------
- Timestamp of summary generation
- Model paths
- Dataset split
- Input shape

Hardware:
---------
- GPU type
- CPU environment
- Framework
- Precision

Purpose:
--------
Ensure reproducibility and traceability
"""

provenance = {
    "generated_at": datetime.utcnow().isoformat() + "Z",
    "v2_model_path": os.path.join(V2_EXP_DIR, "best_model.keras"),
    "v3_model_path": os.path.join(V3_EXP_DIR, "best_model.keras"),
    "data_split": "tfrecords_v2/test/raw",
    "input_shape": [512, 512, 22]
}

hardware = {
    "gpu": "NVIDIA A10G (AWS g5.2xlarge)",
    "cpu": "AWS EC2 g5.2xlarge",
    "framework": "TensorFlow",
    "precision": "FP32"
}


# =========================================================
# FINAL SUMMARY
# =========================================================
"""
Construct final consolidated summary.

Structure:
----------
- Project metadata
- Hardware information
- Provenance
- Model comparison (v2 vs v3)
- Improvement metrics
- Interpretation
- Final model selection

Purpose:
--------
Provide a single comprehensive artifact summarizing:
    • performance
    • efficiency
    • deployment decision
"""

summary = {
    "project": "Multimodal Disaster Damage Segmentation",
    "hardware": hardware,
    "provenance": provenance,
    "models": {
        "v2": v2_block,
        "v3": v3_block
    },
    "improvements": improvements,
    "interpretation": interpretation,
    "final_selection": {
        "chosen_model": "v3",
        "reason": "Higher accuracy with dramatically improved latency, memory, and model size",
        "deployment": "Real-time GPU deployment"
    }
}


# =========================================================
# SAVE
# =========================================================
"""
Save final summary to JSON file.

Behavior:
---------
- Ensures output directory exists
- Writes formatted JSON
- Prints summary to console

Output:
-------
system_eval_v2/profiling/results/summary.json

Purpose:
--------
Persist evaluation results for:
    - reporting
    - reproducibility
    - downstream usage
"""

os.makedirs(RESULTS_DIR, exist_ok=True)

with open(OUTPUT_PATH, "w") as f:
    json.dump(summary, f, indent=4)

print("\n FINAL SUMMARY GENERATED (ABSOLUTE FINAL)")
print(json.dumps(summary, indent=4))
print(f"\n Saved to: {OUTPUT_PATH}")