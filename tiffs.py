
import os as _os, sys as _sys
try:
    import rasterio as _rasterio_probe
    _proj_data = str(_os.path.join(_os.path.dirname(_rasterio_probe.__file__), "proj_data"))
    if _os.path.isfile(_os.path.join(_proj_data, "proj.db")):
        _os.environ["PROJ_DATA"] = _proj_data
        _os.environ["PROJ_LIB"]  = _proj_data
except Exception:
    pass
"""
GEOS-CF NetCDF → Daily-Average GeoTIFF converter
==================================================
Downloads hourly GEOS-CF forecast files from NASA GMAO and produces
24-hour average GeoTIFFs for PM2.5 and NO2 (5 days of forecasts by default).

Default variables:
- PM25_RH35: PM2.5 concentration (μg/m³)
- NO2: Nitrogen dioxide (mol/mol)

Examples
--------
# Full 5-day forecast:
    python3 tiffs.py

# Quick test (download 1 hour):
    python3 tiffs.py --test

# Specify date & hour:
    python3 tiffs.py --init-date 20260312 --init-hour 09

# Keep downloaded .nc4 files:
    python3 tiffs.py --keep-nc4

# Save to custom directory:
    python3 tiffs.py --output-dir my_output

Output filenames
----------------
Daily averages:  geos_cf_<VAR>_<YYYYMMDD>_<HH>z.tif
Test files:      geos_cf_<VAR>_test_<YYYYMMDD>_<HH>z.tif
Example:         geos_cf_PM25_RH35_20260312_09z.tif

Requirements
------------
pip install netCDF4 numpy rasterio requests tqdm
"""

import argparse
import os
import sys
import re
import tempfile
import logging
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from tqdm import tqdm

try:
    import netCDF4 as nc4
except ImportError:
    sys.exit("ERROR: netCDF4 not installed.  Run:  pip install netCDF4")

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
except ImportError:
    sys.exit("ERROR: rasterio not installed.  Run:  pip install rasterio")


# Configuration constants
BASE_URL = "https://portal.nccs.nasa.gov/datashare/gmao/geos-cf/v2/fcst"
COLLECTION = "aqc_tavg_1hr_glo_L1440x721_slv"  # Hourly surface-level data
DEFAULT_VARIABLES = ["PM25_RH35", "NO2"]
FORECAST_LENGTH_HOURS = 120  # ~5 days of forecast from 09z init

# GEOS-CF global grid: 1440×721 at 0.25° resolution
# Longitude: -180 to 180 (1440 cells)
# Latitude: -90 to +90 (721 cells)
GRID_NX = 1440
GRID_NY = 721
GRID_WEST = -180.0
GRID_EAST =  180.0
GRID_SOUTH = -90.0
GRID_NORTH =  90.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# URL building functions

def build_file_url(init_date: str, init_hour: str, valid_dt: datetime) -> str:

    valid_str = valid_dt.strftime("%Y%m%d") + f"_{valid_dt.strftime('%H')}30z"
    filename  = (
        f"GEOS.cf.fcst.{COLLECTION}"
        f".{init_date}_{init_hour}z"
        f"+{valid_str}.R0.nc4"
    )
    year  = init_date[:4]
    month = init_date[4:6]
    day   = init_date[6:8]
    return f"{BASE_URL}/Y{year}/M{month}/D{day}/{filename}"


def latest_init_date_and_hour() -> Tuple[str, str]:
    now_utc = datetime.now(timezone.utc)
    for days_back in range(4):
        candidate = now_utc - timedelta(days=days_back)
        date_str  = candidate.strftime("%Y%m%d")
        # Check if this date has forecast files available
        probe_url = build_file_url(
            date_str, "09",
            datetime(candidate.year, candidate.month, candidate.day, 9, 0,
                     tzinfo=timezone.utc) + timedelta(hours=1)
        )
        try:
            r = requests.head(probe_url, timeout=10)
            if r.status_code == 200:
                log.info("Latest available run: %s 09z", date_str)
                return date_str, "09"
        except requests.RequestException:
            pass
    raise RuntimeError("Could not detect a recent GEOS-CF forecast run.")


# File download utilities

def download_file(url: str, dest: Path, show_progress: bool = True) -> bool:
    try:
        with requests.get(url, stream=True, timeout=60) as resp:
            if resp.status_code != 200:
                log.debug("HTTP %d  %s", resp.status_code, url)
                return False
            total = int(resp.headers.get("content-length", 0))
            with open(dest, "wb") as fh, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=dest.name,
                leave=False,
                disable=not show_progress,
            ) as bar:
                for chunk in resp.iter_content(chunk_size=65536):
                    fh.write(chunk)
                    bar.update(len(chunk))
        return True
    except Exception as exc:
        log.warning("Download failed: %s — %s", url, exc)
        if dest.exists():
            dest.unlink()
        return False


# NetCDF data reading

def extract_variables(nc_path: Path, variables: list[str]) -> dict[str, np.ndarray]:
    out = {}
    with nc4.Dataset(nc_path, "r") as ds:
        for var in variables:
            if var not in ds.variables:
                log.debug("Variable '%s' not found in %s — skipping", var, nc_path.name)
                continue
            raw = ds.variables[var][:]  # Read the variable (may be 3D or 4D)
            arr = np.ma.filled(raw, fill_value=np.nan).squeeze()  # Remove singleton dims
            # If 3D remains (has level dimension), use surface level (index 0)
            if arr.ndim == 3:
                arr = arr[0]
            arr = arr.astype(np.float32)
            if arr.ndim != 2 or arr.shape != (GRID_NY, GRID_NX):
                log.warning(
                    "Unexpected shape %s for '%s' in %s", arr.shape, var, nc_path.name
                )
                continue
            # GEOS-CF stores latitude from S→N; flip to N→S for rasterio
            if "lat" in ds.variables:
                lats = ds.variables["lat"][:]
                if lats[0] < lats[-1]:  # S→N order, flip to N→S
                    arr = np.flipud(arr)
            out[var] = arr
    return out


# Daily averaging and GeoTIFF export

def compute_daily_averages(
    init_date: str,
    init_hour: str,
    forecast_day: datetime,
    variables: list[str],
    tmp_dir: Path,
    keep_nc4: bool = False,
) -> dict[str, np.ndarray]:
    day_start = datetime(
        forecast_day.year, forecast_day.month, forecast_day.day,
        0, 0, tzinfo=timezone.utc
    )

    stacks: dict[str, list[np.ndarray]] = defaultdict(list)
    files_ok = 0

    for hour in range(24):
        valid_dt = day_start + timedelta(hours=hour)
        url      = build_file_url(init_date, init_hour, valid_dt)
        dest     = tmp_dir / url.split("/")[-1]

        if not dest.exists():
            ok = download_file(url, dest, show_progress=True)
            if not ok:
                log.debug("Skipping missing file: %s", url.split("/")[-1])
                continue

        data = extract_variables(dest, variables)
        if data:
            files_ok += 1
            for var, arr in data.items():
                stacks[var].append(arr)

        if not keep_nc4 and dest.exists():
            dest.unlink()

    if files_ok == 0:
        log.warning("No files found for day %s — skipping", forecast_day.strftime("%Y-%m-%d"))
        return {}

    log.info(
        "Day %s: averaged %d/%d hourly files",
        forecast_day.strftime("%Y-%m-%d"),
        files_ok,
        24,
    )

    means: dict[str, np.ndarray] = {}
    for var, arrays in stacks.items():
        if arrays:
            stk = np.stack(arrays, axis=0)       
            means[var] = np.nanmean(stk, axis=0) 
    return means


def write_geotiff(
    data: dict[str, np.ndarray],
    output_path: Path,
    variables: list[str],
    extra_tags: Optional[Dict[str, str]] = None,
) -> None:
    """Write a GeoTIFF file with one band per variable."""
    bands_to_write = [v for v in variables if v in data]
    if not bands_to_write:
        log.warning("No data to write for %s", output_path.name)
        return

    transform = from_bounds(
        GRID_WEST, GRID_SOUTH, GRID_EAST, GRID_NORTH, GRID_NX, GRID_NY
    )
    crs = CRS.from_wkt(
        'GEOGCS["WGS 84",DATUM["WGS_1984",'
        'SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],'
        'UNIT["degree",0.0174532925199433,'
        'AUTHORITY["EPSG","9122"]],'
        'AUTHORITY["EPSG","4326"]]'
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=GRID_NY,
        width=GRID_NX,
        count=len(bands_to_write),
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=np.nan,
        compress="deflate",
        predictor=2,  
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        for band_idx, var in enumerate(bands_to_write, start=1):
            dst.write(data[var], band_idx)
            dst.set_band_description(band_idx, var)
        tags = dict(
            variables=",".join(bands_to_write),
            source="GEOS-CF v2 aqc_tavg_1hr",
            units="see GEOS-CF documentation",
            averaging="24-hour UTC mean",
        )
        if extra_tags:
            tags.update(extra_tags)
        dst.update_tags(**tags)

    log.info("Wrote  %s  (%d bands)", output_path.name, len(bands_to_write))


# Manifest and metadata

def write_manifest(output_dir: Path) -> None:
    """Scan for .tif files and generate layers_manifest.json for the web interface."""
    # Variable metadata: (pollutant key, unit label)
    VAR_META: Dict[str, tuple] = {
        "PM25_RH35":     ("pm25",    "μg/m³"),
        "PM25_RH35_GCC": ("pm25",    "μg/m³"),
        "NO2":           ("no2",     "mol/mol"),
        "O3":            ("o3",      "mol/mol"),
        "CO":            ("co",      "mol/mol"),
        "SO2":           ("so2",     "mol/mol"),
    }

    # Regex patterns for daily and test files
    pat_daily = re.compile(
        r"^geos_cf_(?P<var>[A-Za-z0-9_]+?)_(?P<date>\d{8})_(?P<hour>\d{2})z\.tiff?$"
    )
    pat_test = re.compile(
        r"^geos_cf_(?P<var>[A-Za-z0-9_]+?)_test_(?P<date>\d{8})_(?P<hour>\d{2})z\.tiff?$"
    )

    entries = []
    base_path = "precomputed/pmtiles_output/"

    for f in sorted(output_dir.iterdir()):
        if f.suffix.lower() not in (".tif", ".tiff"):
            continue

        fname = f.name
        is_test = False
        var_raw = None
        date_str = None
        hour_str = None

        m = pat_test.match(fname)
        if m:
            is_test = True
            var_raw  = m.group("var")
            date_str = m.group("date")
            hour_str = m.group("hour")
        else:
            m = pat_daily.match(fname)
            if m:
                var_raw  = m.group("var")
                date_str = m.group("date")
                hour_str = m.group("hour")

        if var_raw:
            pollutant, unit = "default", "ppb"
            for known_var, meta in VAR_META.items():
                safe = re.sub(r"[^A-Za-z0-9_]", "_", known_var)
                if var_raw == safe or var_raw == known_var:
                    pollutant, unit = meta
                    break

            iso_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}" if date_str else None
            tag = " [test]" if is_test else ""
            var_label = var_raw.replace("_", " ")
            name = f"GEOS-CF {var_label} {iso_date}{tag}" if iso_date else fname
        else:

            pollutant, unit = "default", ""
            iso_date = None
            name = fname

        entries.append({
            "file":      fname,
            "path":      base_path + fname,
            "name":      name,
            "pollutant": pollutant,
            "type":      "geotiff",
            "unit":      unit,
            "date":      iso_date,
            "test":      is_test,
        })

    manifest_path = output_dir / "layers_manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump({"layers": entries, "generated": datetime.now(timezone.utc).isoformat()}, fh, indent=2)

    log.info("Manifest written: %s  (%d layers)", manifest_path.name, len(entries))


# Main pipeline

def run(
    init_date: str,
    init_hour: str,
    variables: list[str],
    output_dir: Path,
    keep_nc4: bool,
    days: int,
    tmp_dir_override: Optional[Path],
) -> None:
    """Download and convert GEOS-CF forecast to daily average GeoTIFFs."""
    log.info("=== GEOS-CF → GeoTIFF ===")
    log.info("Forecast init : %s %sz", init_date, init_hour)
    log.info("Variables     : %s", ", ".join(variables))
    log.info("Output dir    : %s", output_dir)
    log.info("Forecast days : %d", days)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert init date/hour to datetime
    init_dt = datetime(
        int(init_date[:4]), int(init_date[4:6]), int(init_date[6:8]),
        int(init_hour), 0, tzinfo=timezone.utc
    )

    # Generate list of forecast days (from init date through N days)
    forecast_days = [
        (init_dt + timedelta(days=d)).replace(hour=0, minute=0, second=0)
        for d in range(days)
    ]

    # Set up temporary directory for .nc4 files
    tmp_ctx = (
        tempfile.TemporaryDirectory(prefix="geos_cf_")
        if not keep_nc4 and tmp_dir_override is None
        else None
    )
    tmp_dir = (
        tmp_dir_override
        if tmp_dir_override is not None
        else (Path(tmp_ctx.name) if tmp_ctx else output_dir / "_nc4_cache")
    )
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        for forecast_day in forecast_days:
            day_str  = forecast_day.strftime("%Y%m%d")
            log.info("--- Processing forecast day %s ---", forecast_day.strftime("%Y-%m-%d"))

            means = compute_daily_averages(
                init_date, init_hour, forecast_day, variables, tmp_dir, keep_nc4
            )
            if not means:
                continue

            # One TIFF per variable per day
            for var in variables:
                if var not in means:
                    continue
                safe_var = re.sub(r"[^A-Za-z0-9_]", "_", var)
                tiff_name = f"geos_cf_{safe_var}_{day_str}_{init_hour}z.tif"
                write_geotiff({var: means[var]}, output_dir / tiff_name, [var])

    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    write_manifest(output_dir)
    log.info("=== Done ===")


# Command-line interface

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download GEOS-CF NetCDF4 files, compute daily averages, export GeoTIFFs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--init-date",
        metavar="YYYYMMDD",
        default=None,
        help="Forecast init date (default: auto-detect latest available run).",
    )
    p.add_argument(
        "--init-hour",
        metavar="HH",
        default="09",
        help="Forecast init hour UTC (default: 09).",
    )
    p.add_argument(
        "--variables",
        nargs="+",
        default=DEFAULT_VARIABLES,
        metavar="VAR",
        help=(
            f"NetCDF variable names to extract (default: {' '.join(DEFAULT_VARIABLES)}). "
            "Run with --list-variables to see all available variables."
        ),
    )
    p.add_argument(
        "--days",
        type=int,
        default=5,
        help="Number of forecast days to produce TIFFs for (default: 5).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("precomputed/pmtiles_output"),
        help="Directory to write GeoTIFFs (default: precomputed/pmtiles_output).",
    )
    p.add_argument(
        "--keep-nc4",
        action="store_true",
        help="Keep downloaded .nc4 files after conversion (saved to --tmp-dir).",
    )
    p.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory for temporary .nc4 downloads (default: system temp dir).",
    )
    p.add_argument(
        "--list-variables",
        action="store_true",
        help="Download one sample file and print all available variable names, then exit.",
    )
    p.add_argument(
        "--test",
        action="store_true",
        help=(
            "Test mode: download a single hourly file (init+1 h) and write one "
            "single-hour TIFF per variable.  Useful for verifying the pipeline "
            "locally without downloading all 120 files."
        ),
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return p.parse_args()


def run_test(
    init_date: str,
    init_hour: str,
    variables: list[str],
    output_dir: Path,
    keep_nc4: bool,
    tmp_dir_override: Optional[Path],
) -> None:
    """Download and write a single hourly GeoTIFF for testing."""
    log.info("=== GEOS-CF → GeoTIFF  [TEST MODE] ===")
    log.info("Forecast init : %s %sz", init_date, init_hour)
    log.info("Variables     : %s", ", ".join(variables))
    log.info("Output dir    : %s", output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    init_dt = datetime(
        int(init_date[:4]), int(init_date[4:6]), int(init_date[6:8]),
        int(init_hour), 0, tzinfo=timezone.utc,
    )
    # Use the first valid hour of the run (init + 1 h)
    valid_dt = init_dt + timedelta(hours=1)
    url      = build_file_url(init_date, init_hour, valid_dt)

    tmp_ctx = (
        tempfile.TemporaryDirectory(prefix="geos_cf_test_")
        if not keep_nc4 and tmp_dir_override is None
        else None
    )
    tmp_dir = (
        tmp_dir_override
        if tmp_dir_override is not None
        else (Path(tmp_ctx.name) if tmp_ctx else output_dir / "_nc4_cache")
    )
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        dest = tmp_dir / url.split("/")[-1]
        log.info("Downloading test file: %s", url.split("/")[-1])
        ok = download_file(url, dest, show_progress=True)
        if not ok:
            log.error("Test download failed. URL: %s", url)
            return

        data = extract_variables(dest, variables)
        if not data:
            log.error("No variables extracted from %s", dest.name)
            return

        valid_date_str = valid_dt.strftime("%Y%m%d")
        valid_hour_str = valid_dt.strftime("%H")

        for var in variables:
            if var not in data:
                log.warning("Variable '%s' not found in test file — skipping", var)
                continue
            safe_var  = re.sub(r"[^A-Za-z0-9_]", "_", var)
            tiff_name = (
                f"geos_cf_{safe_var}_test_{valid_date_str}_{valid_hour_str}z.tif"
            )
            tiff_path = output_dir / tiff_name
            write_geotiff(
                {var: data[var]},
                tiff_path,
                [var],
                extra_tags={
                    "averaging": "single-hour (test mode)",
                    "valid_time": valid_dt.isoformat(),
                },
            )

        if not keep_nc4 and dest.exists():
            dest.unlink()

    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    write_manifest(output_dir)
    log.info("=== Test done — check %s ===", output_dir)


def list_variables(init_date: str, init_hour: str) -> None:
    """Download a sample file and list all available NetCDF variables."""
    init_dt = datetime(
        int(init_date[:4]), int(init_date[4:6]), int(init_date[6:8]),
        int(init_hour), 0, tzinfo=timezone.utc
    )
    sample_dt = init_dt + timedelta(hours=1)
    url = build_file_url(init_date, init_hour, sample_dt)

    print(f"Fetching sample file:\n  {url}\n")
    with tempfile.TemporaryDirectory() as td:
        dest = Path(td) / url.split("/")[-1]
        ok = download_file(url, dest, show_progress=True)
        if not ok:
            print("ERROR: Could not download sample file.")
            return
        with nc4.Dataset(dest, "r") as ds:
            print("Available variables:")
            for name, var in ds.variables.items():
                dims  = ", ".join(var.dimensions)
                units = getattr(var, "units", "—")
                lname = getattr(var, "long_name", "")
                print(f"  {name:<30s}  dims=({dims})  units={units}  {lname}")


if __name__ == "__main__":
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Auto-detect init date if not provided
    if args.init_date is None:
        try:
            args.init_date, args.init_hour = latest_init_date_and_hour()
        except RuntimeError as e:
            sys.exit(f"ERROR: {e}")

    if args.list_variables:
        list_variables(args.init_date, args.init_hour)
        sys.exit(0)

    if args.test:
        run_test(
            init_date=args.init_date,
            init_hour=args.init_hour,
            variables=args.variables,
            output_dir=args.output_dir,
            keep_nc4=args.keep_nc4,
            tmp_dir_override=args.tmp_dir,
        )
        sys.exit(0)

    run(
        init_date=args.init_date,
        init_hour=args.init_hour,
        variables=args.variables,
        output_dir=args.output_dir,
        keep_nc4=args.keep_nc4,
        days=args.days,
        tmp_dir_override=args.tmp_dir,
    )
