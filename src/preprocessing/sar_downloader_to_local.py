import ee
import os
import json
import rasterio
import geemap
from datetime import datetime, timedelta
from typing import List

from src.utils.geo_utils import GeoUtils


class SARDatasetDownloader:
    """
    Downloader for Sentinel-1 SAR (GRD) data using Google Earth Engine.

    This class handles:
    ------------------
    - Event-aware SAR retrieval (pre/post disaster)
    - Multi-tier fallback strategy for missing data
    - Temporal selection of diverse acquisitions
    - Export to local GeoTIFF with metadata

    Key Features:
    -------------
    - Graceful degradation when ideal data is unavailable
    - UTM-aware spatial extraction for accurate geometry
    - Multi-date stacking with polarization support
    - Logging of fallback decisions for transparency

    Fallback Strategy:
    ------------------
    Tier 1: VV+VH, DESCENDING orbit, ≥4 dates  
    Tier 2: VV+VH, ANY orbit, ≥4 dates  
    Tier 3: VV only, ANY orbit, ≥2 dates  

    Used in:
    --------
    Project 1 — SAR preprocessing for segmentation
    """
    
    def __init__(
        self,
        metadata_path: str,
        event_list_json: str,
        polarization: List[str] = ["VV", "VH"],
        min_pixels: int = 512,
        padding_meters: float = 100,
        scale: int = 10,
        is_pre_event: bool = True,
    ):
        """
        Initialize SAR downloader for a specific disaster event.

        Args:
            metadata_path (str): Path to metadata JSON (contains geometry).
            event_list_json (str): Path to event metadata (start/end dates).
            polarization (List[str]): SAR polarizations to use (default: ["VV", "VH"]).
            min_pixels (int): Minimum spatial resolution in pixels.
            padding_meters (float): Padding applied to bounding box.
            scale (int): Spatial resolution (meters per pixel).
            is_pre_event (bool): Whether downloading pre-event or post-event SAR.

        Notes:
            - Automatically determines event name and dates
            - Builds UTM-projected bounding box for accurate spatial operations
        """        
        self.metadata_path = metadata_path
        self.event_name = GeoUtils.get_event_name(metadata_path)

        # --------------------------------------------------
        # Normalize event dates
        # --------------------------------------------------
        raw_start, raw_end = GeoUtils.get_event_dates(
            self.event_name, event_list_json
        )

        def _to_datetime(x):
            if isinstance(x, datetime):
                return x
            if isinstance(x, str):
                return datetime.fromisoformat(x)
            raise TypeError(f"Unsupported date type: {type(x)}")

        self.event_start = _to_datetime(raw_start)
        self.event_end   = _to_datetime(raw_end)

        self.is_pre_event = is_pre_event
        self.prefix = "pre" if is_pre_event else "post"
        self.scale = scale
        self.min_pixels = min_pixels

        # UTM-aware padded bounding box
        self.bbox, self.utm_crs = GeoUtils.get_utm_padded_bounds(
            metadata_path, padding_meters, scale, min_pixels
        )

        # Reporting state
        self.selected_dates = []
        self.fallback_used_days = None
        self.fallback_tier = None
        self.used_polarizations = None
        self.used_orbit = None


    # ------------------------------------------------------------------
    # Sentinel-1 collection helper
    # ------------------------------------------------------------------
    def _get_collection(
        self,
        start_dt: datetime,
        end_dt: datetime,
        polarizations: List[str],
        orbit: str | None,
    ):
        """
        Retrieve filtered Sentinel-1 image collection.

        Filters applied:
        ----------------
        - Spatial bounds (bounding box)
        - Temporal range
        - Instrument mode (IW)
        - Orbit direction (optional)
        - Polarization channels

        Args:
            start_dt (datetime): Start date.
            end_dt (datetime): End date.
            polarizations (List[str]): Required polarizations.
            orbit (str | None): Orbit type ("ASCENDING", "DESCENDING", or None).

        Returns:
            ee.ImageCollection: Filtered SAR collection.
        """        
        col = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(self.bbox)
            .filterDate(
                start_dt.strftime("%Y-%m-%d"),
                end_dt.strftime("%Y-%m-%d"),
            )
            .filter(ee.Filter.eq("instrumentMode", "IW"))
        )

        if orbit is not None:
            col = col.filter(ee.Filter.eq("orbitProperties_pass", orbit))

        for pol in polarizations:
            col = col.filter(
                ee.Filter.listContains("transmitterReceiverPolarisation", pol)
            )

        return col.select(polarizations)


    # ------------------------------------------------------------------
    def pick_most_spread_dates(self, collection, n_dates):
        """
        Select timestamps that maximize temporal spread.

        This ensures selected SAR acquisitions:
        - are not clustered in time
        - capture temporal variability

        Args:
            collection (ee.ImageCollection): Input SAR collection.
            n_dates (int): Number of dates to select.

        Returns:
            List[int]: Selected timestamps (milliseconds since epoch).

        Notes:
            - Uses combinatorial search to maximize time gaps
            - Falls back to empty list if insufficient data
        """
        timestamps = sorted(
            set(collection.aggregate_array("system:time_start").getInfo())
        )
        if len(timestamps) < n_dates:
            return []

        if len(timestamps) == n_dates:
            return timestamps

        from itertools import combinations
        def span(c): return sum(abs(c[i] - c[i - 1]) for i in range(1, len(c)))
        return list(max(combinations(timestamps, n_dates), key=span))


    # ------------------------------------------------------------------
    def build_stack(
        self,
        collection,
        best_dates,
        polarizations,
    ):
        """
        Build multi-temporal SAR stack with normalized values.

        Processing steps:
        -----------------
        - Convert linear SAR to dB scale
        - Clamp values to [-30, +5]
        - Rename bands with prefix + polarization + date
        - Stack all images into single multi-band output

        Args:
            collection (ee.ImageCollection): Filtered SAR collection.
            best_dates (List[int]): Selected timestamps.
            polarizations (List[str]): Polarization channels.

        Returns:
            ee.Image: Multi-band stacked SAR image.

        Notes:
            - Band naming format: {pre/post}_{pol}_{date}
            - Ensures consistent feature representation
        """
        self.selected_dates = [
            datetime.utcfromtimestamp(t / 1000).strftime("%Y-%m-%d")
            for t in best_dates
        ]

        final = collection.filter(
            ee.Filter.inList("system:time_start", best_dates)
        )

        def clean(img):
            dB = img.expression("10 * log10(x)", {"x": img.select(polarizations)})
            dB = dB.clamp(-30, 5).toFloat()

            date_tag = ee.Date(img.get("system:time_start")).format("YYYYMMdd")
            band_names = dB.bandNames().map(
                lambda b: ee.String(self.prefix)
                .cat("_").cat(b).cat("_").cat(date_tag)
            )
            return dB.rename(band_names)

        return final.map(clean).toBands()


    # ------------------------------------------------------------------
    # Fallback v2 (graceful degradation)
    # ------------------------------------------------------------------
    def get_sar_image_with_fallback(
        self,
        initial_days=60,
        max_days=180,
        step_days=15,
    ):
        """
        Retrieve SAR image using progressive fallback strategy.

        Strategy:
        ---------
        Expands temporal window iteratively until sufficient data is found.

        Tries:
            Tier 1 → strict constraints (best quality)
            Tier 2 → relaxed orbit constraint
            Tier 3 → reduced polarization requirement

        Args:
            initial_days (int): Initial temporal window (±days).
            max_days (int): Maximum allowed expansion.
            step_days (int): Increment per iteration.

        Returns:
            ee.Image or None: SAR stack or None if no data found.

        Notes:
            - Logs fallback decisions
            - Guarantees best available data given constraints
        """
        total_days = initial_days

        while total_days <= max_days:

            # --------------------------------------------------
            # Separate PRE and POST windows
            # --------------------------------------------------
            if self.is_pre_event:
                start_dt = self.event_start - timedelta(days=total_days)
                end_dt   = self.event_start

            else:
                start_dt = self.event_end
                end_dt   = self.event_end + timedelta(days=total_days)

            # --------------------------------------------------
            # SAFETY: Prevent overlap (important)
            # --------------------------------------------------
            if self.is_pre_event:
                end_dt = min(end_dt, self.event_start)
            else:
                start_dt = max(start_dt, self.event_end)

            print(
                f"[INFO] {self.event_name} ({self.prefix}) → "
                f"{start_dt.date()} → {end_dt.date()} "
                f"(window={total_days}d)"
            )

            # ---------------- Tier 1 ----------------
            col = self._get_collection(start_dt, end_dt, ["VV", "VH"], "DESCENDING")
            if col.size().getInfo() >= 4:
                dates = self.pick_most_spread_dates(col, 4)
                if dates:
                    self._record_fallback(total_days, 1, ["VV", "VH"], "DESCENDING")
                    return self.build_stack(col, dates, ["VV", "VH"])

            # ---------------- Tier 2 ----------------
            col = self._get_collection(start_dt, end_dt, ["VV", "VH"], None)
            if col.size().getInfo() >= 4:
                dates = self.pick_most_spread_dates(col, 4)
                if dates:
                    self._record_fallback(total_days, 2, ["VV", "VH"], "ANY")
                    return self.build_stack(col, dates, ["VV", "VH"])

            # ---------------- Tier 3 ----------------
            col = self._get_collection(start_dt, end_dt, ["VV"], None)
            if col.size().getInfo() >= 2:
                dates = self.pick_most_spread_dates(col, 2)
                if dates:
                    self._record_fallback(total_days, 3, ["VV"], "ANY")
                    return self.build_stack(col, dates, ["VV"])

            total_days += step_days

        print(
            f"[SKIP] {self.event_name} ({self.prefix}): no SAR after ±{max_days} days"
        )
        return None


    def _record_fallback(self, days, tier, pols, orbit):
        """
        Record fallback configuration used for SAR retrieval.

        Args:
            days (int): Temporal window used.
            tier (int): Fallback tier (1–3).
            pols (List[str]): Polarizations used.
            orbit (str): Orbit type used.

        Notes:
            - Used for reproducibility and debugging
            - Saved in fallback report JSON
        """
        self.fallback_used_days = days
        self.fallback_tier = tier
        self.used_polarizations = pols
        self.used_orbit = orbit

        print(
            f"[OK] SAR found → Tier {tier}, ±{days} days, "
            f"pol={pols}, orbit={orbit}"
        )


    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_to_local(self, image, file_path, nodata_value=-9999):
        """
        Export SAR image to local GeoTIFF and save metadata.

        Outputs:
        --------
        - GeoTIFF image (SAR stack)
        - Dates JSON file
        - Fallback report JSON

        Args:
            image (ee.Image): SAR image to export.
            file_path (str): Output file path.
            nodata_value (int): NoData value for raster.

        Notes:
            - Uses UTM CRS for spatial consistency
            - Ensures reproducibility via metadata logs
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        geemap.ee_export_image(
            ee_object=image.reproject(self.utm_crs, None, self.scale),
            filename=file_path,
            region=self.bbox,
            scale=self.scale,
            crs=self.utm_crs,
            file_per_band=False,
        )

        with rasterio.open(file_path, "r+") as src:
            src.nodata = nodata_value

        # Dates
        with open(file_path.replace(".tif", "_dates.json"), "w") as f:
            json.dump(
                {"event": self.event_name, "dates": self.selected_dates},
                f,
                indent=2,
            )

        # Fallback report
        with open(file_path.replace(".tif", "_fallback_report.json"), "w") as f:
            json.dump(
                {
                    "event": self.event_name,
                    "prefix": self.prefix,
                    "fallback_window_days": self.fallback_used_days,
                    "fallback_tier": self.fallback_tier,
                    "polarizations": self.used_polarizations,
                    "orbit": self.used_orbit,
                    "num_acquisitions": len(self.selected_dates),
                },
                f,
                indent=2,
            )

        print(f"[OK] Download complete → {file_path}")


    # ------------------------------------------------------------------
    def get_sar_image(self):
        """
        Wrapper function to retrieve SAR image.

        Returns:
            ee.Image or None: SAR stack from fallback pipeline.
        """
        return self.get_sar_image_with_fallback()
