import json
from shapely.wkt import loads as load_wkt
from shapely.geometry import box
from shapely.ops import transform
from datetime import datetime, timedelta
from typing import Tuple
import pyproj
import ee


class GeoUtils:
    """
    Utility class for geospatial operations used in Project 1 and Project 2.

    Responsibilities:
    -----------------
    - Extract bounding boxes from metadata (WKT format)
    - Compute SAR temporal windows (pre/post disaster)
    - Handle coordinate system transformations (WGS84 ↔ UTM)
    - Generate spatial regions for Earth Engine exports

    Key Design:
    ----------
    - Uses UTM projection for accurate spatial operations (meters)
    - Converts back to WGS84 for Earth Engine compatibility
    """

    @staticmethod
    def get_coordinate_bounds(metadata_path: str) -> Tuple[float, float, float, float]:
        """
        Extract bounding box coordinates from metadata JSON (WKT geometry).

        This function reads the first available geometry in the metadata file
        and returns its bounding box in WGS84 coordinates.

        Args:
            metadata_path (str): Path to metadata JSON file.

        Returns:
            Tuple[float, float, float, float]:
                (min_lon, min_lat, max_lon, max_lat)

        Raises:
            ValueError: If no geometry is found in the metadata.
        """
        with open(metadata_path) as f:
            data = json.load(f)

        for feature in data["features"]["lng_lat"]:
            wkt_str = feature["wkt"]
            geom = load_wkt(wkt_str)
            return geom.bounds

        raise ValueError("No WKT features found in metadata.")

    @staticmethod
    def orbit_interval_dates(event_start: str, event_end: str, event_type: bool = True,
                             pre_window: int = 60, post_window: int = 60) -> Tuple[datetime, datetime]:
        """
        Compute temporal window for SAR image retrieval.

        This defines the time range used to query Sentinel-1 data:
        - Pre-disaster: window before event start
        - Post-disaster: window after event end

        Args:
            event_start (str): Event start date (YYYY-MM-DD).
            event_end (str): Event end date (YYYY-MM-DD).
            event_type (bool):
                True → pre-disaster window
                False → post-disaster window
            pre_window (int): Days before pre-disaster reference.
            post_window (int): Days after post-disaster reference.

        Returns:
            Tuple[datetime, datetime]: (start_date, end_date)

        Notes:
            - Pre-disaster reference = event_start - 30 days
            - Post-disaster reference = event_end + 3 days
        """
        if event_type:  # pre-event
            if isinstance(event_start, str):
                event_start = datetime.strptime(event_start, "%Y-%m-%d")
            reference = event_start - timedelta(days=30)
            return reference - timedelta(days=pre_window), reference
        else:  # post-event
            if isinstance(event_end, str):
                event_end = datetime.strptime(event_end, "%Y-%m-%d")
            reference = event_end + timedelta(days=3)
            return reference, reference + timedelta(days=post_window)

    @staticmethod
    def get_event_name(metadata_path: str) -> str:
        """
        Extract disaster/event name from metadata JSON.

        Args:
            metadata_path (str): Path to metadata file.

        Returns:
            str: Disaster event identifier (e.g., "hurricane-harvey")
        """
        with open(metadata_path) as f:
            data = json.load(f)
        return data["metadata"]["disaster"]

    @staticmethod
    def get_event_dates(event_name: str, event_list_json_path: str) -> Tuple[str, str]:
        """
        Retrieve event start and end dates from event list JSON.

        Args:
            event_name (str): Disaster event ID.
            event_list_json_path (str): Path to event metadata JSON.

        Returns:
            Tuple[str, str]: (event_start, event_end)

        Raises:
            ValueError: If event is not found in the list.
        """
        with open(event_list_json_path, 'r') as file:
            data = json.load(file)

        for obj in data:
            if obj['id'] == event_name:
                return obj['event_start'], obj['event_end']

        raise ValueError(f"Event '{event_name}' not found in event list.")

    @staticmethod
    def get_utm_padded_bounds(metadata_path: str,
                               padding_meters: float = 100,
                               scale: float = 10,
                               min_pixels: int = 512) -> Tuple[ee.Geometry, str]:
        """
        Compute a padded bounding box in UTM space and convert back to WGS84.

        This function:
        1. Projects bounding box to UTM (meters)
        2. Adds padding
        3. Ensures minimum spatial size (based on pixel resolution)
        4. Converts back to WGS84 for Earth Engine

        Used in:
            Project 1 (segmentation) — ensures full coverage of scene

        Args:
            metadata_path (str): Path to metadata JSON.
            padding_meters (float): Padding around bounding box (meters).
            scale (float): Pixel resolution (meters per pixel).
            min_pixels (int): Minimum size constraint.

        Returns:
            Tuple[ee.Geometry.Rectangle, str]:
                - Earth Engine geometry (WGS84)
                - UTM CRS string (e.g., "EPSG:32615")

        Notes:
            - Ensures spatial consistency across modalities
            - Prevents too-small patches due to sparse geometry
        """
        min_lon, min_lat, max_lon, max_lat = GeoUtils.get_coordinate_bounds(metadata_path)

        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2
        utm_zone = int((center_lon + 180) / 6) + 1
        utm_crs = f"EPSG:{32600 + utm_zone}" if center_lat >= 0 else f"EPSG:{32700 + utm_zone}"

        wgs84 = pyproj.CRS('EPSG:4326')
        utm = pyproj.CRS(utm_crs)
        to_utm = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform
        to_wgs = pyproj.Transformer.from_crs(utm, wgs84, always_xy=True).transform

        original = box(min_lon, min_lat, max_lon, max_lat)
        utm_box = transform(to_utm, original)
        padded = utm_box.buffer(padding_meters, cap_style=3)

        width = padded.bounds[2] - padded.bounds[0]
        height = padded.bounds[3] - padded.bounds[1]
        min_size_m = min_pixels * scale

        if min(width, height) < min_size_m:
            cx = (padded.bounds[0] + padded.bounds[2]) / 2
            cy = (padded.bounds[1] + padded.bounds[3]) / 2
            half = min_size_m / 2
            padded = box(cx - half, cy - half, cx + half, cy + half)

        padded_wgs = transform(to_wgs, padded)
        bounds = padded_wgs.bounds

        return ee.Geometry.Rectangle(bounds), utm_crs

    @staticmethod
    def get_utm_square_bounds(metadata_path: str,
                               scale: float = 10,
                               pixels: int = 512) -> Tuple[ee.Geometry, str]:
        """
        Compute a fixed-size square bounding box in UTM space.

        This function ensures a consistent spatial footprint:
        - Exact size: (pixels × pixels)
        - Resolution: defined by `scale`

        Used in:
            Project 2 (retrieval) — requires uniform patch sizes

        Args:
            metadata_path (str): Path to metadata JSON.
            scale (float): Pixel resolution (meters per pixel).
            pixels (int): Desired image dimension.

        Returns:
            Tuple[ee.Geometry.Rectangle, str]:
                - Earth Engine geometry (WGS84)
                - UTM CRS string

        Notes:
            - Ensures consistent input size for contrastive learning
            - Centered on original bounding box
        """
        min_lon, min_lat, max_lon, max_lat = GeoUtils.get_coordinate_bounds(metadata_path)

        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2
        utm_zone = int((center_lon + 180) / 6) + 1
        utm_crs = f"EPSG:{32600 + utm_zone}" if center_lat >= 0 else f"EPSG:{32700 + utm_zone}"

        wgs84 = pyproj.CRS("EPSG:4326")
        utm = pyproj.CRS(utm_crs)
        to_utm = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform
        to_wgs = pyproj.Transformer.from_crs(utm, wgs84, always_xy=True).transform

        original = box(min_lon, min_lat, max_lon, max_lat)
        utm_box = transform(to_utm, original)

        side_m = pixels * scale

        cx = (utm_box.bounds[0] + utm_box.bounds[2]) / 2
        cy = (utm_box.bounds[1] + utm_box.bounds[3]) / 2

        half = side_m / 2
        square = box(cx - half, cy - half, cx + half, cy + half)

        square_wgs = transform(to_wgs, square)
        bounds = square_wgs.bounds

        return ee.Geometry.Rectangle(bounds), utm_crs