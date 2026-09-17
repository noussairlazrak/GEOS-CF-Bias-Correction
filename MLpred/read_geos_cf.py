# -*- coding: utf-8 -*-
"""
read_geos_cf.py

Standalone module for reading GEOS-CF model data (V1 replay/analysis/forecast
and V2 replay/forecast) from the public S3 zarr stores, with local CSV
caching of the (slow-changing) replay history.
"""

import os
from datetime import datetime

import fsspec
import pandas as pd
import xarray as xr

#: Local cache directory for replay history CSVs (relative to CWD, matching
#: the historical behaviour of ``mlpred.read_geos_cf``).
CACHE_DIR = "GEOS_CF"

# S3 zarr paths
S3_TEMPLATE           = "s3://smce-geos-cf-public/geos-cf-rpl.zarr/"
S3_FORECASTS_TEMPLATE = "s3://smce-geos-cf-public/geos-cf-fcst-latest.zarr/"
S3_REPLAY_TEMPLATE    = "s3://smce-geos-cf-public/geos-cf-ana-latest.zarr"

S3_V2_RPL  = "s3://smce-geos-cf-public/geos-cf-v2-rpl.zarr"
S3_V2_COLS = "s3://smce-geos-cf-public/geos-cf-v2-rpl-cols.zarr"
S3_V2_FCST = "s3://smce-geos-cf-public/geos-cf-v2-fcst-latest.zarr/"

DEFAULT_GASES = ["co", "hcho", "no", "no2", "noy", "o3"]

VVtoPPBV = 1.0e9

#: GEOS-CF replay/analysis/forecast data is hourly; a missing span longer
#: than this multiple of the expected step counts as a gap worth refetching.
GAP_FREQ = "1h"
GAP_MULTIPLIER = 2


def _detect_time_gaps(times, freq=GAP_FREQ, gap_multiplier=GAP_MULTIPLIER):
    """
    Find missing spans in a (possibly unsorted, duplicated) time series.

    Returns a list of ``(gap_start, gap_end)`` tuples, each describing a
    contiguous span of missing timestamps strictly between two known
    points, wherever consecutive points are farther apart than
    ``gap_multiplier`` times the expected sampling frequency.
    """
    t = pd.to_datetime(pd.Series(times)).drop_duplicates().sort_values().reset_index(drop=True)
    if len(t) < 2:
        return []

    step = pd.Timedelta(freq)
    diffs = t.diff()
    gaps = []
    for idx in diffs[diffs > step * gap_multiplier].index:
        gap_start = t.iloc[idx - 1] + step
        gap_end = t.iloc[idx] - step
        if gap_start <= gap_end:
            gaps.append((gap_start, gap_end))
    return gaps


def _fetch_time_range(path, lon, lat, start, end, verbose=True):
    """Fetch a single S3 zarr source restricted to a specific time window."""
    try:
        ds = xr.open_zarr(fsspec.get_mapper(path), consolidated=True)
        sel = {"lon": lon, "lat": lat, "method": "nearest"}
        if "lev" in ds.dims or "lev" in ds.coords:
            sel["lev"] = 1
        ds = ds.sel(**sel).sel(time=slice(start, end))
        df = ds.load().to_dataframe().reset_index()
        df["time"] = pd.to_datetime(df["time"])
        return df
    except Exception as e:
        if verbose:
            print(f"    Warning: could not fetch gap {start} -> {end} from {path}: {e}")
        return None


def _fill_gaps_from_s3(df, source_paths, lon, lat, verbose=True, label="data"):
    """
    Detect gaps in *df* and backfill them by re-requesting each missing
    window from every source in *source_paths* (horizontally merging
    same-time results from multiple sources, e.g. V2's replay + replay_cols).

    Returns ``(df, filled)`` where ``filled`` is True if any gap was
    successfully patched.
    """
    gaps = _detect_time_gaps(df["time"])
    if not gaps:
        return df, False

    if verbose:
        print(f"Detected {len(gaps)} gap(s) in {label} — requesting missing data from S3...")

    fill_frames = []
    for gap_start, gap_end in gaps:
        if verbose:
            print(f"  Gap: {gap_start} -> {gap_end}")
        gap_dfs = [d for d in (
            _fetch_time_range(p, lon, lat, gap_start, gap_end, verbose=verbose)
            for p in source_paths
        ) if d is not None and not d.empty]

        if not gap_dfs:
            if verbose:
                print(f"    Could not backfill gap {gap_start} -> {gap_end} "
                      f"(no data returned from any source — likely a real "
                      f"upstream hole, not just a stale cache).")
            continue

        gap_df = gap_dfs[0]
        for extra in gap_dfs[1:]:
            gap_df = pd.merge(gap_df, extra, on="time", how="outer", suffixes=("", "_x"))
            gap_df = gap_df[[c for c in gap_df.columns if not c.endswith("_x")]]
        fill_frames.append(gap_df)

    if not fill_frames:
        return df, False

    df = (
        pd.concat([df] + fill_frames, ignore_index=True)
        .sort_values("time")
        .drop_duplicates("time", keep="first")
        .reset_index(drop=True)
    )
    if verbose:
        print(f"  Backfilled {label} — now has {len(df)} rows.")
    return df, True


def read_geos_cf(lon, lat, start=None, end=None, version=2, use_cache=True, verbose=True):
    """
    Read GEOS-CF model data from S3 with caching support.

    Caching Strategy:
    - Saves replay data (historical) to CSV: GEOS_CF/loc_{lat}_{lon}_v{version}.csv
    - On subsequent calls: loads cached replay, only fetches new analysis/forecast
    - Significantly reduces S3 read time (replay is ~6+ years of data)
    - Cached replay is checked for internal time gaps (e.g. from a previous
      run that failed partway through); any gap is backfilled from S3 and
      the cache is rewritten. Any gap still present in the final returned
      data is printed as a warning (when ``verbose=True``).

    Parameters
    ----------
    lon, lat : float
        Location coordinates
    start, end : datetime, optional
        Date range filter
    version : int, default=2
        GEOS-CF version (1 or 2)
    use_cache : bool, default=True
        If True, uses cached replay data and only fetches new analysis/forecast
    verbose : bool, default=True
        If True, prints progress messages

    Returns
    -------
    pandas.DataFrame
        Data with species in ppbv and time features

    Examples
    --------
    >>> # First call: reads all data from S3, saves replay to cache
    >>> df = read_geos_cf(lon=-77.0, lat=38.9, version=2)

    >>> # Subsequent calls: loads replay from cache, only fetches new data
    >>> df = read_geos_cf(lon=-77.0, lat=38.9, version=2)  # Much faster!
    """
    # Create cache directory if it doesn't exist
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)

    # filename
    lat_str = f"{lat:.6f}".replace('.', '_').replace('-', 'm')
    lon_str = f"{lon:.6f}".replace('.', '_').replace('-', 'm')
    cache_file = os.path.join(CACHE_DIR, f"loc_{lat_str}_{lon_str}_v{version}.csv")

    # S3 paths
    paths = {
        1: [S3_TEMPLATE, S3_REPLAY_TEMPLATE, S3_FORECASTS_TEMPLATE],
        2: [S3_V2_RPL, S3_V2_COLS, S3_V2_FCST]
    }[version]

    df_replay = None
    cache_needs_resave = False

    # load cached replay data
    if use_cache and os.path.exists(cache_file):
        try:
            if verbose:
                print(f"Loading cached replay data from {cache_file}")
            df_replay = pd.read_csv(cache_file, parse_dates=['time'])
            if verbose:
                print(f"Loaded {len(df_replay)} cached rows (from {df_replay['time'].min()} to {df_replay['time'].max()})")

            # Cached replay can develop internal gaps (e.g. a prior fetch
            # that failed partway through S3). Detect and backfill them
            # from S3 instead of silently trusting the cache.
            replay_paths = [paths[0]] if version == 1 else paths[:2]
            df_replay, cache_needs_resave = _fill_gaps_from_s3(
                df_replay, replay_paths, lon, lat, verbose=verbose, label="cached replay data"
            )
        except Exception as e:
            if verbose:
                print(f"Warning: Could not load cache file: {e}. Will fetch from S3.")
            df_replay = None

    # Read all sources
    dfs = []

    if df_replay is not None:
        dfs.append(df_replay)
        start_index = 2 if version == 2 else 1  # Skip replay (and replay_cols for V2)
        if verbose:
            print(f"Skipping replay data fetch, only fetching recent data...")
    else:
        start_index = 0

    for i, path in enumerate(paths):
        # Skip replay bucket
        if i < start_index:
            continue

        source_name = ["replay", "replay_cols", "forecast"][i] if version == 2 else ["replay", "analysis", "forecast"][i]

        if verbose:
            print(f"Reading {source_name} from S3...")

        try:
            ds = xr.open_zarr(fsspec.get_mapper(path), consolidated=True)
            sel = {"lon": lon, "lat": lat, "method": "nearest"}
            if "lev" in ds.dims or "lev" in ds.coords:
                sel["lev"] = 1
            df = ds.sel(**sel).load().to_dataframe().reset_index()
            df["time"] = pd.to_datetime(df["time"])

            if verbose:
                print(f"Read {len(df)} rows from {source_name}")

            # V2: merge replay and replay_cols horizontally
            if version == 2 and i == 1 and dfs and df_replay is None:
                # Only do horizontal merge if we're reading both replay sources
                dfs[0] = pd.merge(dfs[0], df, on="time", how="outer", suffixes=("", "_x"))
                dfs[0] = dfs[0][[c for c in dfs[0].columns if not c.endswith("_x")]]
                continue

            dfs.append(df)
        except Exception as e:
            if verbose:
                print(f"Error reading {source_name}: {e}")

    if not dfs:
        if verbose:
            print("[ERROR] No data could be retrieved")
        return pd.DataFrame()

    # Merge vertical
    if verbose:
        print(f"Combining {len(dfs)} data sources...")

    df = pd.concat(dfs, ignore_index=True).sort_values("time").drop_duplicates("time", keep="first").reset_index(drop=True)

    # Close any gaps left in the combined series — e.g. a hole inside the
    # upstream S3 replay bucket itself (not just a stale local cache), or a
    # hole straddling the replay/analysis/forecast boundary. Re-request each
    # missing window from every source for this version before giving up.
    df, _filled_full = _fill_gaps_from_s3(df, paths, lon, lat, verbose=verbose, label="combined GEOS-CF series")
    cache_needs_resave = cache_needs_resave or _filled_full

    # Save replay data to cache (fresh fetch, or a gap backfill above)
    if use_cache and (df_replay is None or cache_needs_resave):
        try:
            # V1: save replay only
            # V2: save replay + replay_cols merged
            if version == 1:
                # V2 Replay
                replay_times = df[df['time'] <= (datetime.now() - pd.Timedelta(days=30))]
            else:
                # V2, replay+cols
                replay_times = df[df['time'] <= (datetime.now() - pd.Timedelta(days=5))]

            if len(replay_times) > 0:
                replay_times.to_csv(cache_file, index=False)
                if verbose:
                    print(f"Saved {len(replay_times)} replay rows to {cache_file}")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not save cache file: {e}")

    # Harmonization
    df = df.rename(columns={"t10m": "t", "u10m": "u", "v10m": "v", "pm25_rh35_gcc": "pm25_rh35"})

    # Derived AOD
    if "aod550_sala" in df.columns and "aod550_salc" in df.columns:
        df["aod550_ss"] = df["aod550_sala"] + df["aod550_salc"]

    # Filter dates
    if start:
        df = df[df["time"] >= start]
    if end:
        df = df[df["time"] <= end]

    # Convert to ppbv
    for sp in DEFAULT_GASES + ["so2"]:  # Add so2 to the DEFAULT_GASES list
        if sp in df.columns:
            df[sp] *= VVtoPPBV

    # Add time features
    df["month"] = df["time"].dt.month
    df["hour"] = df["time"].dt.hour
    df["weekday"] = df["time"].dt.weekday

    if verbose:
        print(f"Retrieved {len(df)} total rows from {df['time'].min()} to {df['time'].max()}")
        remaining_gaps = _detect_time_gaps(df["time"])
        if remaining_gaps:
            print(f"Warning: {len(remaining_gaps)} gap(s) remain in the returned data:")
            for gap_start, gap_end in remaining_gaps:
                print(f"    {gap_start} -> {gap_end}")

    return df.reset_index(drop=True)
