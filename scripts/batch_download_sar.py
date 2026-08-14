import ee
import os
import json
import traceback

# -------------------------------------------------------------------
# Earth Engine init
# -------------------------------------------------------------------
ee.Authenticate()
ee.Initialize(project="cv-project1-460420")

from src.preprocessing.sar_downloader_to_local import SARDatasetDownloader


# ===================================================================
def batch_download_sar(
    metadata_dir: str,
    event_list_json: str,
    is_pre_event: bool,
    export_dir: str,
):
    """
    Batch download SAR images for a given dataset split and event type.

    This function:
    ----------------
    - Iterates through metadata files (one per patch/event)
    - Downloads SAR imagery using SARDatasetDownloader
    - Applies fallback strategies when necessary
    - Exports SAR stacks to local disk
    - Tracks detailed metrics and logs outcomes

    Args:
        metadata_dir (str): Directory containing metadata JSON files.
        event_list_json (str): Path to global event metadata (start/end dates).
        is_pre_event (bool): Whether to download pre-disaster or post-disaster SAR.
        export_dir (str): Output directory for SAR GeoTIFF files.

    Outputs:
        - GeoTIFF SAR stacks (multi-band)
        - Batch summary JSON file containing:
            • number of successful downloads
            • skipped events
            • fallback usage statistics
            • polarization and orbit usage

    Notes:
        - Uses fallback tiers defined in SARDatasetDownloader
        - Designed for large-scale dataset processing
        - Robust to missing or invalid SAR data

    Run Example:
        export DATA_ROOT=/mnt/ebs-data/cv_project1_new/data
        python batch_download_sar.py
    """
    prefix = "pre" if is_pre_event else "post"

    if not os.path.exists(metadata_dir):
        print(f"[WARN] Metadata directory does not exist: {metadata_dir}")
        return

    metadata_files = sorted(f for f in os.listdir(metadata_dir) if f.endswith(".json"))

    print(f"\n=== {prefix.upper()} — {len(metadata_files)} metadata files ===")

    summary = {
        "total": len(metadata_files),
        "downloaded": 0,
        "skipped": 0,
        "fallback_used": 0,
        "tier_counts": {"1": 0, "2": 0, "3": 0},
        "polarization_counts": {},
        "orbit_counts": {},
        "events": {},
    }

    for idx, filename in enumerate(metadata_files, start=1):
        metadata_path = os.path.join(metadata_dir, filename)
        base_name = os.path.splitext(filename)[0]

        print("\n---------------------------------------------------")
        print(f"[{idx}/{len(metadata_files)}] {base_name}")
        print("---------------------------------------------------")

        downloader = SARDatasetDownloader(
            metadata_path=metadata_path,
            event_list_json=event_list_json,
            polarization=["VV", "VH"],
            is_pre_event=is_pre_event,
        )

        try:
            sar_img = downloader.get_sar_image()

            if sar_img is None:
                print("[SKIP] No SAR image generated.")
                summary["skipped"] += 1
                summary["events"][base_name] = {
                    "status": "skipped",
                }
                continue

            os.makedirs(export_dir, exist_ok=True)
            out_tif = os.path.join(export_dir, f"{base_name}_{prefix}_sar.tif")

            downloader.export_to_local(
                image=sar_img,
                file_path=out_tif,
            )

            # ---------------- Metrics ----------------
            summary["downloaded"] += 1

            if downloader.fallback_used_days is not None:
                summary["fallback_used"] += 1

            tier = str(downloader.fallback_tier)
            pols = ",".join(downloader.used_polarizations)
            orbit = downloader.used_orbit

            summary["tier_counts"][tier] += 1
            summary["polarization_counts"][pols] = (
                summary["polarization_counts"].get(pols, 0) + 1
            )
            summary["orbit_counts"][orbit] = (
                summary["orbit_counts"].get(orbit, 0) + 1
            )

            summary["events"][base_name] = {
                "status": "downloaded",
                "fallback_days": downloader.fallback_used_days,
                "fallback_tier": downloader.fallback_tier,
                "polarizations": downloader.used_polarizations,
                "orbit": downloader.used_orbit,
            }

        except Exception as e:
            print(f"[ERROR] Failed on {base_name}: {e}")
            traceback.print_exc()
            summary["skipped"] += 1
            summary["events"][base_name] = {
                "status": "error",
                "error": str(e),
            }

    # --------------------------------------------------------------
    # Save batch summary
    # --------------------------------------------------------------
    summary_path = os.path.join(export_dir, f"{prefix}_batch_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===================================================")
    print(f"[SUMMARY] {prefix.upper()}")
    print(f"  Downloaded     : {summary['downloaded']}")
    print(f"  Skipped        : {summary['skipped']}")
    print(f"  Used fallback  : {summary['fallback_used']}")
    print(f"  Tier counts    : {summary['tier_counts']}")
    print(f"  Polarizations  : {summary['polarization_counts']}")
    print(f"  Orbits         : {summary['orbit_counts']}")
    print(f"  Summary saved  : {summary_path}")
    print("===================================================\n")


# ===================================================================
def run():
    """
    Main entry point for batch SAR downloading across dataset splits.

    This function:
    ----------------
    - Defines dataset root and event metadata paths
    - Iterates over dataset splits (train, hold, test)
    - Processes both pre- and post-disaster SAR data
    - Constructs appropriate input/output directories
    - Calls batch_download_sar for each configuration

    Workflow:
    ---------
    For each split:
        For each event type (pre, post):
            1. Load metadata files
            2. Download SAR data with fallback handling
            3. Export GeoTIFF outputs
            4. Save summary metrics

    Notes:
        - Assumes consistent directory structure:
            dataset_root/
                ├── train/
                ├── hold/
                ├── test/
        - Metadata directory must contain pre_disaster / post_disaster subfolders
        - Designed for reproducible batch processing

    Run Example:
        export DATA_ROOT=/mnt/ebs-data/cv_project1_new/data
        python batch_download_sar.py
    """

    DATA_ROOT = os.getenv("DATA_ROOT", ".")

    dataset_root = os.path.join(DATA_ROOT, "sar_data")
    event_list_json = os.path.join(dataset_root, "xview2_events.json")

    splits = ["train", "hold", "test"]

    for split in splits:
        for is_pre_event in [True, False]:

            prefix = "pre" if is_pre_event else "post"

            export_dir = os.path.join(dataset_root, split, f"sar_{prefix}")
            os.makedirs(export_dir, exist_ok=True)

            metadata_dir = os.path.join(
                DATA_ROOT,
                "rgb_data",
                split,
                "labels",
                f"{prefix}_disaster",
            )

            print("\n===================================================")
            print(f" SPLIT: {split.upper()}   EVENT: {prefix.upper()} ")
            print(f" Metadata dir: {metadata_dir}")
            print(f" Output dir:   {export_dir}")
            print("===================================================\n")

            batch_download_sar(
                metadata_dir=metadata_dir,
                event_list_json=event_list_json,
                is_pre_event=is_pre_event,
                export_dir=export_dir,
            )


# ===================================================================
if __name__ == "__main__":
    run()