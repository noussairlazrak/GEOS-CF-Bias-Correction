# -*- coding: utf-8 -*-
"""
GEOS-FP CNN Module

Fetches MERRA-2 CNN PM2.5 forecasts from the AERONET GeoJSON endpoint and
merges them with GEOS-CF data. CNN data is the primary source; GEOS-CF fills
any gaps.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests

from MLpred import mlpred, funcs
from MLpred.s3_manager import S3Manager


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

GEOJSON_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "GEOS_FP_CNN",
)
GEOJSON_S3_PREFIX = "GEOS_FP_CNN/"
DEFAULT_S3_BUCKET: Optional[str] = os.environ.get("GEOS_FP_CNN_S3_BUCKET")

_GEOJSON_CACHE: dict[str, pd.DataFrame] = {}


def _default_s3_manager() -> Optional[S3Manager]:
    if DEFAULT_S3_BUCKET:
        return S3Manager(bucket_name=DEFAULT_S3_BUCKET)
    return None


def _geojson_date_str(date: datetime) -> str:
    return date.strftime("%Y%m%d")


def _geojson_local_path(date_str: str) -> str:
    os.makedirs(GEOJSON_CACHE_DIR, exist_ok=True)
    return os.path.join(GEOJSON_CACHE_DIR, f"{date_str}_forecast.geojson")


def _parse_geojson(geojson: dict) -> pd.DataFrame:
    rows = []
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        coords = feature.get("geometry", {}).get("coordinates", [None, None])
        try:
            base_date = pd.to_datetime(props["UTC_DATE"])
        except Exception:
            continue
        for hour_offset, conc_col, aqi_col in _3HR_SLOTS:
            rows.append({
                "time":          base_date + timedelta(hours=hour_offset),
                "pm25_conc_cnn": props.get(conc_col),
                "pm25_aqi":      props.get(aqi_col),
                "daily_aqi":     props.get("DAILY_AQI"),
                "Station":       props.get("Station"),
                "Site_Name":     props.get("Site_Name"),
                "lat":           coords[1],
                "lon":           coords[0],
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"])
    return df


def _fetch_geojson_for_date(
    date: datetime,
    silent: bool = True,
    s3_manager: Optional[S3Manager] = None,
    s3_bucket: Optional[str] = None,
    s3_prefix: str = GEOJSON_S3_PREFIX,
) -> pd.DataFrame:
    if s3_manager is None:
        s3_manager = _default_s3_manager()

    date_str   = _geojson_date_str(date)
    local_path = _geojson_local_path(date_str)

    # In-memory
    if date_str in _GEOJSON_CACHE and not _GEOJSON_CACHE[date_str].empty:
        if not silent:
            print(f"Using cached forecast for {date_str}.")
        return _GEOJSON_CACHE[date_str]

    # Local
    if os.path.exists(local_path):
        file_size = os.path.getsize(local_path)
        if file_size < 100:
            print(f"Local forecast file for {date_str} not completed ({file_size} B)")
            try:
                os.remove(local_path)
            except OSError:
                pass
        else:
            try:
                with open(local_path, "r") as _f:
                    cached_geojson = json.load(_f)
                df = _parse_geojson(cached_geojson)
                if not df.empty:
                    print(f"Loaded forecast from local disk: {date_str} ({len(df):,} rows).")
                    _GEOJSON_CACHE[date_str] = df
                    return df
                print(f"Disk file for {date_str} -> no station data")
                os.remove(local_path)
            except Exception as exc:
                print(f"Could not read local forecast file for {date_str} ({exc}).")
                try:
                    os.remove(local_path)
                except OSError:
                    pass
    else:
        if not silent:
            print(f"No local forecast file found for {date_str}.")

    # Remote
    url = MERRA2CNN_GEOJSON_TEMPLATE.format(date=date_str)
    print(f"Downloading GEOS-FP forecast for {date_str} …")
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        geojson = response.json()
        df = _parse_geojson(geojson)
        if df.empty:
            print(f"Download succeeded - no station data for {date_str}.")
            return pd.DataFrame()
        print(f"Download complete: {len(df):,} rows for {date_str}.")
    except Exception as exc:
        print(f"Could not download forecast for {date_str}: {exc}")
        return pd.DataFrame()

    _GEOJSON_CACHE[date_str] = df
    try:
        with open(local_path, "w") as _f:
            json.dump(geojson, _f)
        print(f"Saved to {local_path}.")
        if s3_manager is not None:
            _bucket = s3_bucket or s3_manager.bucket_name
            s3_key  = f"{s3_prefix.rstrip('/')}/{os.path.basename(local_path)}"
            if s3_manager.upload_file(file_path=local_path, s3_key=s3_key, bucket=_bucket):
                print(f"Uploaded to S3: s3://{_bucket}/{s3_key}")
            else:
                print(f"S3 upload failed for {s3_key}.")
    except Exception as exc:
        print(f"Could not save forecast to disk: {exc}")

    return df


def _find_station(
    df: pd.DataFrame,
    site: Optional[str],
    lat: Optional[float],
    lon: Optional[float],
    silent: bool = True,
) -> pd.DataFrame:
    if df.empty:
        return df

    if site:
        site_lower = site.strip().lower()

        mask = df["Site_Name"].astype(str).str.lower() == site_lower
        if mask.any():
            if not silent:
                print(f"Matched station by name: '{df.loc[mask, 'Site_Name'].iloc[0]}'")
            return df[mask]

        mask = df["Station"].astype(str).str.lower() == site_lower
        if mask.any():
            if not silent:
                print(f"Matched station by ID: '{df.loc[mask, 'Station'].iloc[0]}'")
            return df[mask]

        mask = (
            df["Site_Name"].astype(str).str.lower().str.contains(site_lower, regex=False, na=False) |
            df["Station"].astype(str).str.lower().str.contains(site_lower, regex=False, na=False)
        )
        if mask.any():
            if not silent:
                print(f"Matched station by partial name: {df.loc[mask, 'Site_Name'].unique().tolist()}")
            return df[mask]

        if not silent:
            print(f"No station named '{site}' found — falling back to nearest coordinates.")

    if lat is not None and lon is not None:
        stations = df[["Station", "lat", "lon"]].drop_duplicates("Station").copy()
        stations["dist"] = ((stations["lat"] - lat) ** 2 + (stations["lon"] - lon) ** 2) ** 0.5
        best = stations.loc[stations["dist"].idxmin(), "Station"]
        filtered = df[df["Station"] == best]
        if not silent:
            print(f"Using nearest station: '{filtered['Site_Name'].iloc[0]}' ({best})")
        return filtered

    return df


def _reshape_to_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """ GeoJSON slot times to the nearest 3h boundary and deduplicate."""
    if df.empty:
        return df
    df = df.copy()
    df["time"] = df["time"].dt.round("3h")
    return df.sort_values("time").drop_duplicates(subset="time").reset_index(drop=True)


def load_geojson_all_locations(
    date: Optional[datetime] = None,
    silent: bool = True,
    s3_manager: Optional[S3Manager] = None,
    s3_bucket: Optional[str] = None,
    s3_prefix: str = GEOJSON_S3_PREFIX,
) -> pd.DataFrame:
    """Return the full global GeoJSON for *date* (defaults to today) as a DataFrame."""
    if date is None:
        date = datetime.today()
    return _fetch_geojson_for_date(
        date, silent=silent,
        s3_manager=s3_manager, s3_bucket=s3_bucket, s3_prefix=s3_prefix,
    )


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
    Return an hourly PM2.5 forecast DataFrame merged from GEOS-FP CNN and GEOS-CF.

    Fetch order: today's file → yesterday's file → GEOS-CF only.
    Each 3-hourly CNN slot is forward-filled to cover the full 3-hour window.
    GEOS-CF (hourly) fills any remaining gaps.
    """
    end_date   = datetime.today() + timedelta(days=5)
    start_date = datetime.today() - timedelta(days=frequency)
    base_date  = geojson_date if geojson_date is not None else datetime.today()
    all_data   = pd.DataFrame()

    if not skip_geosfp:
        for day_offset, label in [(0, "current day"), (1, "-1day")]:
            attempt_date = base_date - timedelta(days=day_offset)
            print(f"Searching for GEOS-FP forecast: {label} ({attempt_date.strftime('%Y-%m-%d')}) …")

            full_df = _fetch_geojson_for_date(
                attempt_date, silent=silent,
                s3_manager=s3_manager, s3_bucket=s3_bucket, s3_prefix=s3_prefix,
            )

            if not full_df.empty:
                if day_offset > 0:
                    print(f"Using {label}'s forecast")
                station_df = _find_station(full_df, site, lat, lon, silent=silent)
                cnn_3h     = _reshape_to_timeseries(station_df)

                if not cnn_3h.empty:
                    hourly_index = pd.date_range(
                        start=cnn_3h["time"].min(),
                        end=cnn_3h["time"].max() + timedelta(hours=2),
                        freq="1h",
                    )
                    all_data = (
                        cnn_3h.set_index("time")
                        .reindex(hourly_index)
                        .ffill(limit=2)
                        .reset_index()
                        .rename(columns={"index": "time"})
                    )
                    if not silent:
                        print(f"Expanded {len(cnn_3h)} 3-hourly: {len(all_data)} hourly rows.")
                else:
                    all_data = cnn_3h

                if lat is None and not all_data.empty:
                    lat = float(all_data["lat"].iloc[0])
                if lon is None and not all_data.empty:
                    lon = float(all_data["lon"].iloc[0])
                break
        else:
            print("No GEOS-FP forecast available")

    if all_data.empty:
        print("No GEOS-FP CNN data available. GEOS-CF Initiated")
        time_range = pd.date_range(start=start_date, end=end_date, freq="1h")
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

    if not silent:
        print("Fetching GEOS-CF data …")

    geos_cf = mlpred.read_geos_cf(lon=lon, lat=lat, start=start_date, end=end_date, version=2)

    def _strip_tz(df: pd.DataFrame, col: str = "time") -> pd.DataFrame:
        if df[col].dt.tz is not None:
            df = df.copy()
            df[col] = df[col].dt.tz_convert("UTC").dt.tz_localize(None)
        return df

    all_data = _strip_tz(all_data)
    if not geos_cf.empty:
        geos_cf = _strip_tz(geos_cf)

    if not silent:
        cnn_valid = all_data["pm25_conc_cnn"].notna().sum()
        print(f"CNN: {len(all_data)} hourly rows ({cnn_valid} with data) | GEOS-CF: {len(geos_cf)} rows.")

    merg = funcs.merge_dataframes([all_data, geos_cf], "time", resample="1h", how="outer")

    if not silent:
        print(f"Merged to {len(merg)} hourly rows.")

    core_cols = [c for c in ["pm25_rh35", "no2", "o3", "t", "rh"] if c in merg.columns]
    if core_cols:
        has_geos_cf = merg[core_cols].notna().any(axis=1)
        has_cnn     = merg["pm25_conc_cnn"].notna() if "pm25_conc_cnn" in merg.columns else pd.Series(False, index=merg.index)
        before  = len(merg)
        merg    = merg[has_geos_cf | has_cnn].reset_index(drop=True)
        dropped = before - len(merg)
        if not silent and dropped:
            print(f"Removed {dropped} empty rows")

    if "pm25_conc_cnn" in merg.columns and "pm25_rh35" in merg.columns:
        missing_mask = merg["pm25_conc_cnn"].isna()
        if missing_mask.any():
            merg.loc[missing_mask, "pm25_conc_cnn"] = merg.loc[missing_mask, "pm25_rh35"]
            if not silent:
                print(f"Filled {missing_mask.sum()} hours from GEOS-CF PM2.5 ")
        merg["pm25source"] = np.where(missing_mask, "GEOS-CF", "GEOS-FP CNN")

    species_map = {"PM2.5": "pm25_rh35", "NO2": "no2", "O3": "o3"}
    merg = funcs.calculate_nowcast(merg, species_columns=species_map, avg_hours={"NO2": 3, "O3": 1})

    if "pm25_aqi" in merg.columns and "PM25_NowCast_AQI" in merg.columns:
        missing_aqi = merg["pm25_aqi"].isna()
        if missing_aqi.any():
            merg.loc[missing_aqi, "pm25_aqi"] = merg.loc[missing_aqi, "PM25_NowCast_AQI"]

    for axis in ("lat", "lon"):
        if f"{axis}_x" in merg.columns:
            merg[axis] = merg[f"{axis}_x"].combine_first(merg.get(f"{axis}_y"))
            merg.drop(columns=[c for c in (f"{axis}_x", f"{axis}_y") if c in merg.columns], inplace=True)

    if not silent:
        cnn_rows = (merg.get("pm25source") == "GEOS-FP CNN").sum()
        gcf_rows = (merg.get("pm25source") == "GEOS-CF").sum()
        print(f"Final forecast: {cnn_rows} hours from GEOS-FP CNN, {gcf_rows} hours from GEOS-CF.")

    return funcs.convert_times_column(merg, "time", lat, lon)
