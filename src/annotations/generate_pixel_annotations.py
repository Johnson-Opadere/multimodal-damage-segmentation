import os
import json
import shapely.wkt
import geopandas as gpd
from tqdm import tqdm


def convert_xview_json_to_pixel_geojson(
    json_path: str,
    output_path: str
):
    """
    Convert xView2 annotation JSON to pixel-space GeoJSON.

    This function:
    ----------------
    - Reads xView2 annotation JSON (WKT polygons)
    - Converts polygons into Shapely geometries
    - Preserves original pixel coordinate system (0–1023)
    - Saves output as GeoJSON for downstream processing

    Args:
        json_path (str): Path to input annotation JSON file
        output_path (str): Path to output GeoJSON file

    Notes:
        - Coordinates remain in pixel space (NOT georeferenced)
        - Invalid or empty geometries are skipped
        - Output CRS is None (non-geographic)
    """

    with open(json_path) as f:
        data = json.load(f)

    features = []

    for item in data.get("features", {}).get("xy", []):
        poly = shapely.wkt.loads(item["wkt"])

        if poly.is_empty or not poly.is_valid:
            continue

        props = item.get("properties", {})

        features.append({
            "geometry": poly,
            "properties": props
        })

    if not features:
        print(f"[WARN] No valid polygons in {os.path.basename(json_path)}")
        return

    # Pixel-space CRS (non-geographic)
    gdf = gpd.GeoDataFrame.from_features(features, crs=None)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gdf.to_file(output_path, driver="GeoJSON")

    print(f"[OK] Saved pixel annotation → {output_path}")


def batch_convert(
    input_dir,
    output_dir,
    suffix="_post_disaster.json"
):
    """
    Batch convert xView2 JSON annotations to pixel-space GeoJSON.

    Args:
        input_dir (str): Directory containing input JSON files
        output_dir (str): Directory to save GeoJSON outputs
        suffix (str): File suffix filter (default: post-disaster labels)

    Notes:
        - Processes only files matching suffix
        - Maintains one-to-one file mapping (JSON → GeoJSON)
    """
    files = [f for f in os.listdir(input_dir) if f.endswith(suffix)]
    os.makedirs(output_dir, exist_ok=True)

    for fname in tqdm(files, desc="Generating pixel annotations"):
        in_path = os.path.join(input_dir, fname)
        out_name = fname.replace(".json", ".geojson")
        out_path = os.path.join(output_dir, out_name)

        convert_xview_json_to_pixel_geojson(in_path, out_path)


if __name__ == "__main__":
    """
    Main execution for generating pixel-space annotations.

    Workflow:
    ---------
    - Processes train, hold, and test splits
    - Converts post-disaster annotations to GeoJSON
    - Saves outputs into dataset_root structure

    Notes:
        - Assumes consistent directory structure
        - Only post-disaster annotations are processed

    Run Example:
    ------------
    export DATA_ROOT=/mnt/ebs-data/cv_project1_new/data
    python src/annotations/generate_pixel_annotations.py
    """

    DATA_ROOT = os.getenv("DATA_ROOT", ".")

    dataset_root = os.path.join(DATA_ROOT, "rgb_data")
    BASE_INPUT = os.path.join(DATA_ROOT, "rawData")

    # TRAIN
    batch_convert(
        input_dir=os.path.join(BASE_INPUT, "train", "labels", "post_disaster"),
        output_dir=os.path.join(dataset_root, "train", "annot_post")
    )

    # HOLD
    batch_convert(
        input_dir=os.path.join(BASE_INPUT, "hold", "labels", "post_disaster"),
        output_dir=os.path.join(dataset_root, "hold", "annot_post")
    )

    # TEST
    batch_convert(
        input_dir=os.path.join(BASE_INPUT, "test", "labels", "post_disaster"),
        output_dir=os.path.join(dataset_root, "test", "annot_post")
    )