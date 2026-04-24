# -*- coding: utf-8 -*-
"""
GEOS-FP CNN Module

Module for fetching and processing GEOS-FP CNN (MERRA-2) data combined with GEOS-CF data.
Data is sourced from a single dated GeoJSON endpoint that contains all global locations.

Author: Noussair Lazrak
"""

import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests

# Local imports
from MLpred import mlpred, funcs
from MLpred.s3_manager import S3Manager



MERRA2CNN = "https://aeronet.gsfc.nasa.gov/cgi-bin/web_print_air_quality_index"
MERRA2CNN_GEOJSON_TEMPLATE = (
    "https://aeronet.gsfc.nasa.gov/data_push/AQI/output_DoS_geoJSON/{date}_forecast.geojson"
)


_3HR_SLOTS = [
    (1,  "3HR_PM_CONC_CNN(130)",  "3HR_AQI(130)"),
    (4,  "3HR_PM_CONC_CNN(430)",  "3HR_AQI(430)"),
    (7,  "3HR_PM_CONC_CNN(730)",  "3HR_AQI(730)"),
    (10, "3HR_PM_CONC_CNN(1030)", "3HR_AQI(1030)"),
    (13, "3HR_PM_CONC_CNN(1330)", "3HR_AQI(1330)"),
    (16, "3HR_PM_CONC_CNN(1630)", "3HR_AQI(1630)"),
    (19, "3HR_PM_CONC_CNN(1930)", "3HR_AQI(1930)"),
    (22, "3HR_PM_CONC_CNN(2230)", "3HR_AQI(2230)"),
]

# Local disk cache
GEOJSON_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "GEOS_FP_CNN",
)

# Default S3 prefix
GEOJSON_S3_PREFIX = "GEOS_FP_CNN/"

# Default S3 bucket 
DEFAULT_S3_BUCKET: Optional[str] = os.environ.get("GEOS_FP_CNN_S3_BUCKET")


def _default_s3_manager() -> Optional[S3Manager]:
    """Return a default S3Manager when DEFAULT_S3_BUCKET is configured, else None."""
    if DEFAULT_S3_BUCKET:
        return S3Manager(bucket_name=DEFAULT_S3_BUCKET)
    return None

# in-memory GeoJSON cache 
_GEOJSON_CACHE: dict[str, pd.DataFrame] = {}


def _geojson_date_str(date: datetime) -> str:
    """Return the date string used in the GeoJSON filename (YYYYMMDD)."""
    return date.strftime("%Y%m%d")


def _geojson_local_path(date_str: str) -> str:
    """Return the local CSV path for a given YYYYMMDD date string."""
    os.makedirs(GEOJSON_CACHE_DIR, exist_ok=True)
    return os.path.join(GEOJSON_CACHE_DIR, f"{date_str}_forecast.csv")


def _parse_geojson(geojson: dict) -> pd.DataFrame:
    """Parse a GeoJSON FeatureCollection into a long-format DataFrame."""
    rows = []
    for feature in geojson.get("features", []):
        props  = feature.get("properties", {})
        coords = feature.get("geometry", {}).get("coordinates", [None, None])
        feat_lon, feat_lat = coords[0], coords[1]

        try:
            base_date = pd.to_datetime(props["UTC_DATE"])
        except Exception:
            continue

        station   = props.get("Station")
        site_name = props.get("Site_Name")
        daily_aqi = props.get("DAILY_AQI")

        for hour_offset, conc_col, aqi_col in _3HR_SLOTS:
            rows.append({
                "time":          base_date + timedelta(hours=hour_offset),
                "pm25_conc_cnn": props.get(conc_col),
                "pm25_aqi":      props.get(aqi_col),
                "daily_aqi":     daily_aqi,
                "Station":       station,
                "Site_Name":     site_name,
                "lat":           feat_lat,
                "lon":           feat_lon,
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"])
    return df


def _fetch_geojson_for_date(
    date: datetime,
    silent: bool = True,
    s3_manager: Optional["S3Manager"] = None,
    s3_bucket: Optional[str] = None,
    s3_prefix: str = GEOJSON_S3_PREFIX,
) -> pd.DataFrame:
    """
    Return the parsed global forecast GeoJSON for *date* as a long-format DataFrame.

    Lookup order
    ------------
    1. In-memory cache (``_GEOJSON_CACHE``) – free, same session.
    2. Local CSV file in ``GEOJSON_CACHE_DIR`` – fast, survives restarts.
    3. Remote GeoJSON endpoint – downloaded once, then saved to disk and memory.

    After a fresh download the CSV is also uploaded to S3 when *s3_manager* is
    provided (defaults to ``_default_s3_manager()`` which reads
    ``GEOS_FP_CNN_S3_BUCKET`` from the environment).

    Columns: ``time``, ``pm25_conc_cnn``, ``pm25_aqi``, ``daily_aqi``,
             ``Station``, ``Site_Name``, ``lat``, ``lon``.

    Parameters
    ----------
    date : datetime
    silent : bool
    s3_manager : S3Manager or None, optional
        Pass an explicit manager to override the default.  Pass ``False`` to
        disable S3 upload entirely even when a default bucket is configured.
    s3_bucket : str, optional
        Target S3 bucket (uses ``s3_manager.bucket_name`` when omitted).
    s3_prefix : str
        S3 key prefix (folder) for the uploaded CSV.

    Returns
    -------
    pd.DataFrame  (empty if all sources fail)
    """
    # Resolve default S3 manager lazily so callers can pass False to opt out
    if s3_manager is None:
        s3_manager = _default_s3_manager()
    date_str   = _geojson_date_str(date)
    local_path = _geojson_local_path(date_str)

    # 1. In-memory
    if date_str in _GEOJSON_CACHE:
        if not silent:
            print(f"in-memory cache for {date_str}")
        return _GEOJSON_CACHE[date_str]

    # 2. Disk 
    if os.path.exists(local_path):
        if not silent:
            print(f"loading from disk: {local_path}")
        try:
            df = pd.read_csv(local_path, parse_dates=["time"])
            _GEOJSON_CACHE[date_str] = df
            # Still upload to S3 if manager is provided (file may not be there yet)
            if s3_manager is not None:
                _bucket = s3_bucket or s3_manager.bucket_name
                s3_key = f"{s3_prefix.rstrip('/')}/{os.path.basename(local_path)}"
                uploaded = s3_manager.upload_file(
                    file_path=local_path,
                    s3_key=s3_key,
                    bucket=_bucket,
                )
                if uploaded:
                    region = getattr(s3_manager, "region_name", "us-east-1") or "us-east-1"
                    s3_url = f"https://{_bucket}.s3.{region}.amazonaws.com/{s3_key}"
                    print(f"Uploaded to S3: s3://{_bucket}/{s3_key}")
                    print(f"    URL: {s3_url}")
                else:
                    print(f"S3 upload failed for {s3_key}")
            return df
        except Exception as exc:
            if not silent:
                print(f"  disk load failed ({exc}), re-downloading …")

    # 3. Remote 
    url = MERRA2CNN_GEOJSON_TEMPLATE.format(date=date_str)
    if not silent:
        print(f"downloading: {url}")

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        geojson = response.json()
    except Exception as exc:
        if not silent:
            print(f"  download failed ({exc})")
        return pd.DataFrame()

    df = _parse_geojson(geojson)

    if df.empty:
        _GEOJSON_CACHE[date_str] = df
        return df


    try:
        df.to_csv(local_path, index=False)
        if not silent:
            print(f"  saved to {local_path}")
        # Upload to S3 if a manager was supplied
        if s3_manager is not None:
            _bucket = s3_bucket or s3_manager.bucket_name
            s3_key = f"{s3_prefix.rstrip('/')}/{os.path.basename(local_path)}"
            uploaded = s3_manager.upload_file(
                file_path=local_path,
                s3_key=s3_key,
                bucket=_bucket,
            )
            if uploaded:
                region = getattr(s3_manager, "region_name", "us-east-1") or "us-east-1"
                s3_url = f"https://{_bucket}.s3.{region}.amazonaws.com/{s3_key}"
                print(f"Uploaded to S3: s3://{_bucket}/{s3_key}")
                print(f"    URL: {s3_url}")
            else:
                print(f"S3 upload failed for {s3_key}")
    except Exception as exc:
        if not silent:
            print(f"  could not save to disk ({exc})")

    _GEOJSON_CACHE[date_str] = df
    if not silent:
        print(f"  {len(df)} 3-hourly records across {df['Station'].nunique()} stations")
    return df


def _find_station(df: pd.DataFrame,
                  site: Optional[str],
                  lat: Optional[float],
                  lon: Optional[float],
                  silent: bool = True) -> pd.DataFrame:
    """
    Filter *df* (all-location GeoJSON frame) to the single best-matching station.

    Matching priority
    -----------------
    1. Exact ``Site_Name`` match in the JSON (case-insensitive)
    2. Exact ``Station`` ID match
    3. ``Site_Name`` or ``Station`` substring match (case-insensitive)
    4. Nearest lat/lon if *site* is None or nothing matched above
    """
    if df.empty:
        return df

    if site:
        site_lower = site.strip().lower()

        # 1. Name Match
        mask_exact = df["Site_Name"].astype(str).str.lower() == site_lower
        if mask_exact.any():
            filtered = df[mask_exact]
            if not silent:
                print(f"  exact Site_Name match: '{filtered['Site_Name'].iloc[0]}'")
            return filtered

        # 2. Station ID
        mask_station = df["Station"].astype(str).str.lower() == site_lower
        if mask_station.any():
            filtered = df[mask_station]
            if not silent:
                print(f"  exact Station ID match: '{filtered['Station'].iloc[0]}'")
            return filtered

        # 3. Substring match
        mask_sub = (
            df["Site_Name"].astype(str).str.lower().str.contains(site_lower, regex=False, na=False) |
            df["Station"].astype(str).str.lower().str.contains(site_lower, regex=False, na=False)
        )
        if mask_sub.any():
            filtered = df[mask_sub]
            if not silent:
                matched_names = filtered["Site_Name"].unique().tolist()
                print(f"  substring match(es): {matched_names}")
            return filtered

        if not silent:
            print(f"  '{site}' not found in Site_Name or Station; falling back to nearest lat/lon")

    if lat is not None and lon is not None:
        unique_stations = df[["Station", "lat", "lon"]].drop_duplicates("Station").copy()
        unique_stations["dist"] = (
            (unique_stations["lat"] - lat) ** 2 +
            (unique_stations["lon"] - lon) ** 2
        ) ** 0.5
        best_station = unique_stations.loc[unique_stations["dist"].idxmin(), "Station"]
        filtered = df[df["Station"] == best_station]
        if not silent:
            name = filtered["Site_Name"].iloc[0]
            print(f"  nearest station: '{name}' ({best_station})")
        return filtered

    if not silent:
        print("  no site filter applied; returning all locations")
    return df


def _reshape_to_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate and sort a station-filtered GeoJSON frame by time.

    GeoJSON slots arrive at UTC offsets 1h, 4h, 7h … (i.e. 01:00, 04:00 …)
    while GEOS-CF data uses a standard 0h, 3h, 6h … grid.  Rounding to the
    nearest 3-hour boundary here ensures that both sources land in the same
    resample bin so that ``merge_dataframes(..., resample="3h")`` can combine
    them without introducing spurious NaN rows.
    """
    if df.empty:
        return df
    df = df.copy()
    df["time"] = df["time"].dt.round("3h")
    return (
        df
        .sort_values("time")
        .drop_duplicates(subset="time")
        .reset_index(drop=True)
    )


#  Public API

def load_geojson_all_locations(
    date: Optional[datetime] = None,
    silent: bool = True,
    s3_manager: Optional[S3Manager] = None,
    s3_bucket: Optional[str] = None,
    s3_prefix: str = GEOJSON_S3_PREFIX,
) -> pd.DataFrame:
    """
    Download (or return from cache) the full global GeoJSON for *date*.

    Parameters
    ----------
    date : datetime, optional
        Forecast date to load.  Defaults to today.
    silent : bool
    s3_manager : S3Manager, optional
        If provided, uploads the CSV to S3 after the first download.
    s3_bucket : str, optional
        Target S3 bucket.
    s3_prefix : str
        S3 key prefix (folder) for the uploaded CSV.

    Returns
    -------
    pd.DataFrame
        Long-format frame with all stations and 3-hourly PM2.5 / AQI values.
        Columns: ``time``, ``pm25_conc_cnn``, ``pm25_aqi``, ``daily_aqi``,
        ``Station``, ``Site_Name``, ``lat``, ``lon``.
    """
    if date is None:
        date = datetime.today()
    return _fetch_geojson_for_date(date, silent=silent, s3_manager=s3_manager, s3_bucket=s3_bucket, s3_prefix=s3_prefix)


def read_geos_fp_cnn(
    site: Optional[str] = None,
    frequency: int = 10,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    silent: bool = True,
    skip_geosfp: bool = False,
    geojson_date: Optional[datetime] = None,
    s3_manager: Optional[S3Manager] = None,
    s3_bucket: Optional[str] = None,
    s3_prefix: str = GEOJSON_S3_PREFIX,
) -> pd.DataFrame:
    """
    Fetch and merge GEOS-FP CNN (MERRA-2) data with GEOS-CF data.

    The MERRA-2 CNN data is now sourced from a **single GeoJSON endpoint**
    (``MERRA2CNN_GEOJSON_TEMPLATE``) that contains all global locations for a
    given forecast date.  The full GeoJSON is loaded once into memory
    (``_GEOJSON_CACHE``) and then filtered to the requested site, so repeated
    calls for different sites on the same date do not re-download the file.

    Parameters
    ----------
    site : str, optional
        Station ID or Site_Name substring to select (case-insensitive).
        If None, the station nearest to *lat*/*lon* is used.
    frequency : int, optional
        Number of days back from today used as the GEOS-CF start date
        (default: 10).  The GeoJSON forecast always covers the 3-day window
        centred on *geojson_date*.
    lat : float, optional
        Latitude of the location (used for nearest-station fallback and
        GEOS-CF extraction).
    lon : float, optional
        Longitude of the location (used for nearest-station fallback and
        GEOS-CF extraction).
    silent : bool, optional
        Suppress print output (default: True).
    skip_geosfp : bool, optional
        If True, skip the GeoJSON fetch entirely (default: False).
    geojson_date : datetime, optional
        Forecast date for the GeoJSON file.  Defaults to today.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with columns:
        ``time``, ``pm25_conc_cnn``, ``pm25_aqi``, ``daily_aqi``,
        ``Station``, ``Site_Name``, ``lat``, ``lon``,
        GEOS-CF variables (``pm25_rh35``, ``no2``, ``o3``, …),
        NowCast AQI columns, and local-time column added by
        ``funcs.convert_times_column``.

    Examples
    --------
    >>> df = read_geos_fp_cnn(site="Nairobi", lat=-1.23, lon=36.82, silent=False)
    >>> print(df.columns.tolist())

    Notes
    -----
    - The GeoJSON contains 3-day windows (yesterday, today, tomorrow relative
      to the server-side forecast run).
    - Missing MERRA-2 CNN values are back-filled from GEOS-CF ``pm25_rh35``.
    - NowCast and AQI calculations are applied after merging.
    - Time is converted to local time based on *lat*/*lon*.
    """
    end_date   = datetime.today() + timedelta(days=5)
    start_date = datetime.today() - timedelta(days=frequency)


    all_data = pd.DataFrame()

    if not skip_geosfp:
        if geojson_date is None:
            geojson_date = datetime.today()

        if not silent:
            print(f"Loading GeoJSON for {geojson_date.strftime('%Y-%m-%d')} …")

        full_df = _fetch_geojson_for_date(
            geojson_date,
            silent=silent,
            s3_manager=s3_manager,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
        )

        if not full_df.empty:
            station_df = _find_station(full_df, site, lat, lon, silent=silent)
            all_data   = _reshape_to_timeseries(station_df)

            # station lat/lon
            if lat is None and not all_data.empty:
                lat = float(all_data["lat"].iloc[0])
            if lon is None and not all_data.empty:
                lon = float(all_data["lon"].iloc[0])
        else:
            if not silent:
                print("GeoJSON returned no data.")


    if all_data.empty:
        if not silent:
            print("No MERRA-2 CNN data; creating NaN placeholder series.")
        time_range = pd.date_range(start=start_date, end=end_date, freq="3h")
        all_data = pd.DataFrame({
            "time":          time_range,
            "pm25_conc_cnn": np.nan,
            "pm25_aqi":      np.nan,
            "daily_aqi":     np.nan,
            "Station":       None,
            "Site_Name":     None,
            "lat":           lat,
            "lon":           lon,
        })

    # Fetch matching GEOS-CF data 
    if not silent:
        print("Requesting GEOS-CF …")

    geos_cf = mlpred.read_geos_cf(
        lon=lon,
        lat=lat,
        start=start_date,
        end=end_date,
        version=2,
    )
    # Merge MERRA-2 CNN and GEOS-CF 
    merg = funcs.merge_dataframes([all_data, geos_cf], "time", resample="3h", how="outer")
    if not silent:
        print("Merged columns:", merg.columns.tolist())

    # Drop rows where ALL core GEOS-CF variables are NaN (stale cache ghost rows)
    core_cols = [c for c in ["pm25_rh35", "no2", "o3", "t", "rh"] if c in merg.columns]
    if core_cols:
        before = len(merg)
        merg = merg.dropna(subset=core_cols, how="all").reset_index(drop=True)
        dropped = before - len(merg)
        if not silent and dropped:
            print(f"Dropped {dropped} rows with no GEOS-CF data (likely stale cache rows)")

    # Fill missing GEOS FP CNN values with GEOS-CF PM2.5 (pm25_rh35)
    filled_counts: dict = {}

    if "pm25_conc_cnn" in merg.columns and "pm25_rh35" in merg.columns:
        missing_mask = merg["pm25_conc_cnn"].isna()
        n_missing = missing_mask.sum()
        if n_missing > 0:
            merg.loc[missing_mask, "pm25_conc_cnn"] = merg.loc[missing_mask, "pm25_rh35"]
            filled_counts["pm25_conc_cnn"] = n_missing
            if not silent:
                print(f"Filled {n_missing} pm25_conc_cnn values with pm25_rh35")
        merg["pm25source"] = np.where(missing_mask, "GEOS-CF", "GEOS-FP CNN")

    # NowCast and AQI calculations
    species_map = {"PM2.5": "pm25_rh35", "NO2": "no2", "O3": "o3"}
    avg_hours   = {"NO2": 3, "O3": 1}
    merg = funcs.calculate_nowcast(merg, species_columns=species_map, avg_hours=avg_hours)

    if "pm25_aqi" in merg.columns and "PM25_NowCast_AQI" in merg.columns:
        missing_mask = merg["pm25_aqi"].isna()
        n_missing = missing_mask.sum()
        if n_missing > 0:
            merg.loc[missing_mask, "pm25_aqi"] = merg.loc[missing_mask, "PM25_NowCast_AQI"]
            filled_counts["pm25_aqi"] = n_missing
            if not silent:
                print(f"Filled {n_missing} pm25_aqi values with PM25_NowCast_AQI")

    # Cleaning
    for axis in ("lat", "lon"):
        if f"{axis}_x" in merg.columns:
            merg[axis] = merg[f"{axis}_x"].combine_first(merg.get(f"{axis}_y"))
            merg.drop(columns=[c for c in (f"{axis}_x", f"{axis}_y") if c in merg.columns], inplace=True)

    if not silent:
        if filled_counts:
            print(f"Fill summary: {filled_counts}")
        else:
            print("No fill needed.")
        print("Final columns:", merg.columns.tolist())

    # Convert to local time 
    result = funcs.convert_times_column(merg, "time", lat, lon)
    return result
