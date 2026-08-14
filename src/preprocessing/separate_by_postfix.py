import os
import shutil
from collections import defaultdict
import json


def get_invalid_files(source_folder):
    """
    Identify files with missing coordinate information.
    Returns a set of invalid event names.
    """
    invalid_files = set()

    if not os.path.isdir(source_folder):
        print(f"Error: {source_folder} is not a valid directory.")
        return invalid_files

    for filename in os.listdir(source_folder):
        filepath = os.path.join(source_folder, filename)

        if not os.path.isfile(filepath):
            continue

        parts = filename.split('_', 2)
        if len(parts) != 3:
            continue

        name_part, postfix_ext1, _ = parts
        combined_event_name = f"{name_part}_{postfix_ext1}"

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            if len(data.get("features", {}).get("lng_lat", [])) == 0:
                invalid_files.add(combined_event_name)

        except Exception as e:
            print(f"Skipping file {filename}: {e}")
            continue

    return invalid_files


def separate_by_postfix(source_folder, invalid_files):
    """
    Separate files into subfolders based on postfix while skipping invalid files.
    """
    postfix_files = defaultdict(list)

    if not os.path.isdir(source_folder):
        print(f"Error: {source_folder} is not a valid directory.")
        return

    for filename in os.listdir(source_folder):
        filepath = os.path.join(source_folder, filename)

        if not os.path.isfile(filepath):
            continue

        parts = filename.split('_', 2)
        if len(parts) != 3:
            continue

        name_part, postfix_ext1, postfix_ext2 = parts
        combined_event_name = f"{name_part}_{postfix_ext1}"

        if combined_event_name in invalid_files:
            continue

        postfix = os.path.splitext(postfix_ext2)[0]
        postfix_files[postfix].append(filename)

    # Move files into postfix-based folders
    for postfix, files in postfix_files.items():
        subfolder = os.path.join(source_folder, postfix)
        os.makedirs(subfolder, exist_ok=True)

        for file in files:
            src_path = os.path.join(source_folder, file)
            dst_path = os.path.join(subfolder, file)
            shutil.move(src_path, dst_path)

    print("Files successfully separated into subfolders by postfix.")