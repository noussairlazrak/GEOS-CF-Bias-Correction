# -*- coding: utf-8 -*-
"""
pandora.py

Standalone module for reading and parsing Pandora instrument data files.
Supports NO2 and O3 pollutants.

Author: Noussair Lazrak
"""

import hashlib
import os
import re
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests

#: Root directory
OBS_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "OBS",
)

DEFAULT_CACHE_HOURS: int = 24


def _site_from_url(url: str) -> str:
    """
    Extract a short, filesystem-safe site name from a Pandora file URL.

    The Pandora URL structure is:
        .../data.pandonia-global-network.org/<Site>/Pandora<N>s<N>/L2/<file>
    so the first path segment after the host is the site name (e.g. ``Agam``).
    Falls back to the first 10 chars of the URL MD5 if parsing fails.
    """
    try:
        from urllib.parse import urlparse
        parts = [p for p in urlparse(url).path.split("/") if p]
        for part in parts:
            if part and not part.startswith("Pandora") and not part.startswith("L"):
                return re.sub(r'[^\w\-]', '_', part)
    except Exception:
        pass
    return hashlib.md5(url.encode()).hexdigest()[:10]


def _cache_path(pollutant: str, location: Optional[str] = None, url: Optional[str] = None) -> str:
    """
    Return the local CSV path for a Pandora observation.

    Filename format: ``<Site>_<pollutant>.csv``  (e.g. ``Agam_no2.csv``).
    *location* overrides the site derived from *url*.
    """
    os.makedirs(OBS_CACHE_DIR, exist_ok=True)
    if location:
        site = re.sub(r'[^\w\-]', '_', location.strip())
    elif url:
        site = _site_from_url(url)
    else:
        raise ValueError("Either 'location' or 'url' must be provided to _cache_path")
    filename = f"{site}_{pollutant.lower()}.csv"
    return os.path.join(OBS_CACHE_DIR, filename)


def _is_cache_fresh(path: str, hours: int) -> bool:
    """Return True if *path* exists and was modified within the last *hours*."""
    if not os.path.exists(path):
        return False
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
    return age < timedelta(hours=hours)


def extract_metadata(content: str) -> dict:
    """
    Extract site metadata from a Pandora file's header section.

    Parameters
    ----------
    content : str
        Raw text content of the Pandora file.

    Returns
    -------
    dict
        Dictionary with keys: ``location_name``, ``latitude``, ``longitude``.
    """
    location_name = re.search(r'Full location name:\s*(.+)', content).group(1).strip()
    latitude = float(re.search(r'Location latitude \[deg\]:\s*([-\d.]+)', content).group(1))
    longitude = float(re.search(r'Location longitude \[deg\]:\s*([-\d.]+)', content).group(1))
    return {
        'location_name': location_name,
        'latitude': latitude,
        'longitude': longitude,
    }


def convert_no2_mol_m3_to_ppbv(
    no2_mol_m3: pd.Series,
    temperature_k: pd.Series,
    pressure_mbar: pd.Series,
) -> pd.Series:
    """
    Convert NO2 concentration from mol/m³ to ppbv using the ideal gas law.

    Parameters
    ----------
    no2_mol_m3 : array-like
        NO2 concentration in mol/m³.
    temperature_k : array-like
        Ambient temperature in Kelvin.
    pressure_mbar : array-like
        Ambient pressure in millibars.

    Returns
    -------
    pd.Series
        NO2 concentration in ppbv.
    """
    pressure_atm = pressure_mbar / 1013.25
    return no2_mol_m3 * (24.45 * 1e9) / (0.0821 * temperature_k * pressure_atm)


def read_pandora(
    url: str,
    pollutant: str = 'no2',
    cache: bool = True,
    cache_hours: int = DEFAULT_CACHE_HOURS,
    silent: bool = False,
) -> pd.DataFrame:
    """
    Fetch and parse a Pandora instrument data file.

    The function downloads the file at *url*, strips the header, and
    extracts per-measurement columns for the requested *pollutant*.
    Filtering for quality / physical-range constraints is applied
    automatically.

    A local CSV cache is maintained under ``OBS_CACHE_DIR``.  If the cached
    file is younger than *cache_hours* it is returned directly without
    hitting the remote server.

    Parameters
    ----------
    url : str
        Direct URL to the Pandora ASCII data file.
    pollutant : {'no2', 'o3'}
        Pollutant to extract.  Defaults to ``'no2'``.
    cache : bool
        Enable local caching (default: ``True``).
    cache_hours : int
        Age threshold in hours below which the local cache is considered
        fresh (default: ``24``).
    silent : bool
        Suppress cache-related print messages (default: ``False``).

    Returns
    -------
    pd.DataFrame
        Columns: ``time``, ``lat``, ``lon``, ``value``, ``location``.

        * For NO2: ``value`` is in ppbv (scaled by 1/40 after unit
          conversion).
        * For O3:  ``value`` is the total column in Dobson Units.

    Raises
    ------
    ValueError
        If *pollutant* is not ``'no2'`` or ``'o3'``.
    requests.HTTPError
        If the remote file cannot be downloaded.
    """
    cache_file = _cache_path(pollutant, url=url)

    if cache and _is_cache_fresh(cache_file, cache_hours):
        age_h = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))).total_seconds() / 3600
        if not silent:
            print(f"Using cached data ({age_h:.1f}h old): {cache_file}")
        return pd.read_csv(cache_file, parse_dates=["time"])

    if not silent:
        print(f"Downloading {pollutant.upper()} data for '{_site_from_url(url)}'")

    response = requests.get(url)
    response.raise_for_status()
    content = response.text

    metadata = extract_metadata(content)

    separator = "---------------------------------------------------------------------------------------"
    data_start_index = content.index(separator) + len(separator)

    df = pd.read_csv(
        url,
        skiprows=data_start_index,
        header=None,
        sep=r'\s+',
        encoding='ISO-8859-1',
        on_bad_lines='skip',
    )
    df = df.iloc[:, :69]
    df.columns = range(df.shape[1])

    df['time'] = pd.to_datetime(df[0], format="%Y%m%dT%H%M%S.%fZ")

    # ------------------------------------------------------------------ NO2 --
    if pollutant.lower() == 'no2':
        result_df = pd.DataFrame({
            'time':                          df['time'],
            'no2_surface_conc_mol_m3':       df[55],
            'no2_surface_conc_uncertainty':  df[56],
            'no2_surface_conc_index':        df[57],
            'no2_heterogeneity_flag':        df[58],
            'no2_stratospheric_column':      df[59],
            'no2_tropospheric_column':       df[61],
            'no2_layer1_height_km':          df[67],
            'pressure_mbar':                 df[13],
            'temperature_k':                 df[14],
            'solar_zenith_angle':            df[3],
            'quality_flag_no2':              df[52],
            'integration_time_ms':           df[30],
            'wavelength_shift_nm':           df[28],
        })
        result_df['lat'] = float(metadata['latitude'])
        result_df['lon'] = float(metadata['longitude'])
        result_df['location'] = metadata['location_name']
        result_df['value'] = df[55] * 1e8

        numeric_columns = [
            'no2_surface_conc_mol_m3', 'no2_surface_conc_uncertainty',
            'pressure_mbar', 'temperature_k',
        ]
        result_df = result_df[result_df['no2_surface_conc_mol_m3'] != -9e99]
        result_df[numeric_columns] = result_df[numeric_columns].apply(
            pd.to_numeric, errors='coerce'
        )

        result_df['value'] = convert_no2_mol_m3_to_ppbv(df[55], df[14], df[13]) / 40

        result_df = result_df[
            (result_df['value'] > 0) &
            (result_df['value'] < 100)
        ]

    # ------------------------------------------------------------------ O3 ---
    elif pollutant.lower() == 'o3':
        result_df = pd.DataFrame({
            'time':                                  df['time'],
            'fractional_day':                        df[1],
            'measurement_duration':                  df[2],
            'solar_zenith_angle':                    df[3],
            'solar_azimuth':                         df[4],
            'lunar_zenith_angle':                    df[5],
            'lunar_azimuth':                         df[6],
            'rms_unweighted':                        df[7],
            'rms_weighted':                          df[8],
            'expected_rms_unweighted':               df[9],
            'expected_rms_weighted':                 df[10],
            'pressure_mbar':                         df[11],
            'data_processing_type_index':            df[12],
            'calibration_file_version':              df[13],
            'calibration_validity_start':            df[14],
            'mean_measured_value':                   df[15],
            'wavelength_effective_temp_C':           df[16],
            'avg_residual_stray_light_percent':      df[17],
            'retrieved_wavelength_shift_L1':         df[18],
            'retrieved_total_wavelength_shift':      df[19],
            'retrieved_resolution_change_percent':   df[20],
            'integration_time_ms':                   df[21],
            'num_bright_count_cycles':               df[22],
            'filterwheel1_position':                 df[23],
            'filterwheel2_position':                 df[24],
            'atmospheric_variability_percent':       df[25],
            'aerosol_opt_depth_start':               df[26],
            'aerosol_opt_depth_center':              df[27],
            'aerosol_opt_depth_end':                 df[28],
            'L1_data_quality_flag':                  df[29],
            'L1_quality_sum_DQ1':                    df[30],
            'L1_quality_sum_DQ2':                    df[31],
            'L2Fit_data_quality_flag':               df[32],
            'L2Fit_quality_sum_DQ1':                 df[33],
            'L2Fit_quality_sum_DQ2':                 df[34],
            'quality_flag_o3':                       df[35],
            'L2_quality_sum_DQ1_ozone':              df[36],
            'L2_quality_sum_DQ2_ozone':              df[37],
            'o3_total_column':                       df[38],
            'o3_column_uncertainty':                 df[39],
            'o3_structured_uncertainty':             df[40],
            'o3_common_uncertainty':                 df[41],
            'o3_total_uncertainty':                  df[42],
            'o3_rms_uncertainty':                    df[43],
            'temperature_k':                         df[44],
            'o3_effective_temp_indep_uncertainty':   df[45],
            'o3_effective_temp_structured_uncertainty': df[46],
            'o3_effective_temp_common_uncertainty':  df[47],
            'o3_effective_temp_total_uncertainty':   df[48],
            'direct_ozone_air_mass_factor':          df[49],
            'ozone_air_mass_factor_uncertainty':     df[50],
            'diffuse_correction_percent':            df[51],
        })
        result_df['lat'] = float(metadata['latitude'])
        result_df['lon'] = float(metadata['longitude'])
        result_df['location'] = metadata['location_name']
        result_df['value'] = df[38]

        numeric_columns = [
            'o3_total_column', 'o3_column_uncertainty',
            'pressure_mbar', 'temperature_k',
        ]
        result_df[numeric_columns] = result_df[numeric_columns].apply(
            pd.to_numeric, errors='coerce'
        )

        result_df = result_df[
            (result_df['o3_total_column'] != -9e99) &
            (result_df['quality_flag_o3'].isin([0, 1, 2, 10, 11, 12])) &
            (result_df['o3_total_column'] >= 0) &
            (result_df['aerosol_opt_depth_center'] >= 0) &
            (result_df['o3_total_column'] <= 500)
        ]

    else:
        raise ValueError(
            f"Unsupported pollutant '{pollutant}'. Choose 'no2' or 'o3'."
        )

    result_df = result_df[["time", "lat", "lon", "value", "location"]]

    # Save to local cache
    if cache:
        try:
            result_df.to_csv(cache_file, index=False)
            if not silent:
                print(f"Cached {len(result_df)} rows → {cache_file}")
        except Exception as exc:
            if not silent:
                print(f"Warning: could not write cache ({exc})")

    return result_df
