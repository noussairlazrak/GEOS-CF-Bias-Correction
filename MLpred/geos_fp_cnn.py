# -*- coding: utf-8 -*-
"""
GEOS-FP CNN Module

Module for fetching and processing GEOS-FP CNN (MERRA-2) data combined with GEOS-CF data.

Author: Noussair Lazrak
"""

import io
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Local imports
from MLpred import mlpred, funcs


# Constants
MERRA2CNN = "https://aeronet.gsfc.nasa.gov/cgi-bin/web_print_air_quality_index"


def read_geos_fp_cnn(
    base_url: str = MERRA2CNN,
    site: Optional[str] = None,
    frequency: int = 30,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    silent: bool = True,
    skip_geosfp: bool = False
) -> pd.DataFrame:
    """
    Fetch and merge GEOS-FP CNN (MERRA-2) data with GEOS-CF data.
    
    This function retrieves PM2.5 concentration predictions from the MERRA-2 CNN model
    and merges them with GEOS-CF atmospheric data for a specified location.
    
    Parameters
    ----------
    base_url : str, optional
        Base URL for the MERRA-2 CNN API (default: AERONET air quality index endpoint)
    site : str, optional
        Site identifier for the MERRA-2 CNN API
    frequency : int, optional
        Number of days of historical data to fetch (default: 30)
    lat : float, optional
        Latitude of the location
    lon : float, optional
        Longitude of the location
    silent : bool, optional
        If True, suppress print statements (default: True)
    skip_geosfp : bool, optional
        If True, skip fetching GEOS-FP data and only retrieve GEOS-CF (default: False)
        
    Returns
    -------
    pd.DataFrame
        Merged DataFrame containing:
        - time: Timestamp of the observation
        - pm25_conc_cnn: PM2.5 concentration from CNN model
        - pm25_aqi: Air Quality Index for PM2.5
        - Station: Station identifier
        - Site_Name: Station name
        - GEOS-CF variables (pm25_rh35, no2, o3, etc.)
        - NowCast and AQI calculations
        
    Examples
    --------
    >>> loc = read_geos_fp_cnn(
    ...     site="site_name_or_id",
    ...     lat= latitude_value,
    ...     lon= longitude_value,
    ...     frequency=7,
    ...     silent=False
    ... )
    >>> print(loc.columns.tolist())

    Notes
    -----
    - The function fetches 3-hourly data from MERRA-2 CNN
    - Missing MERRA-2 values are filled with corresponding GEOS-CF values
    - NowCast and AQI calculations are performed on the merged data
    - Time is converted to local time based on lat/lon
    """
    end_date = datetime.today() + timedelta(days=5)
    start_date = end_date - timedelta(days=frequency)
    all_data = pd.DataFrame()

    # MERRA2 fetch
    if not skip_geosfp:
        for n in range(frequency + 3):
            date = start_date + timedelta(days=n)
            
            # URL construction
            if site:
                url = f"{base_url}?year={date.year}&month={date.month}&day={date.day}&site={site}"
            else:
                url = f"{base_url}?year={date.year}&month={date.month}&day={date.day}"

            if not silent:
                print(f"Fetching {date.strftime('%Y-%m-%d')}")

            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Error check
                if "Error:" in response.text or "No data available" in response.text:
                    if not silent:
                        print(f"No data {date.strftime('%Y-%m-%d')}")
                    continue
                
                # HTML parsing
                soup = BeautifulSoup(response.text, 'html.parser')
                text = soup.body.get_text() if soup.body else response.text
                
                # CSV extraction
                lines = text.strip().split('\n')
                
                # Header search
                csv_start_idx = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith('Station,'):
                        csv_start_idx = i
                        break
                
                # CSV join
                csv_text = '\n'.join(lines[csv_start_idx:])
                
                if not csv_text.strip():
                    if not silent:
                        print(f"Empty CSV {date.strftime('%Y-%m-%d')}")
                    continue
                
                df = pd.read_csv(io.StringIO(csv_text))
                if df.empty:
                    if not silent:
                        print("Empty DataFrame")
                    continue
                
                # Site filter
                if site and 'Station' in df.columns:
                    df = df[df['Station'].astype(str).str.contains(site, case=False, na=False) | 
                            df['Site_Name'].astype(str).str.contains(site, case=False, na=False)]
                    if df.empty:
                        if not silent:
                            print(f"No data for {site} on {date.strftime('%Y-%m-%d')}")
                        continue

                reshaped_data = []
                for _, row in df.iterrows():
                    try:
                        base_date = pd.to_datetime(row['UTC_DATE'])
                        for hour_offset, conc_col, aqi_col in zip(
                            [1, 4, 7, 10, 13, 16, 19, 22],
                            ['3HR_PM_CONC_CNN(130)', '3HR_PM_CONC_CNN(430)', '3HR_PM_CONC_CNN(730)', 
                             '3HR_PM_CONC_CNN(1030)', '3HR_PM_CONC_CNN(1330)', '3HR_PM_CONC_CNN(1630)', 
                             '3HR_PM_CONC_CNN(1930)', '3HR_PM_CONC_CNN(2230)'],
                            ['3HR_AQI(130)', '3HR_AQI(430)', '3HR_AQI(730)', '3HR_AQI(1030)', 
                             '3HR_AQI(1330)', '3HR_AQI(1630)', '3HR_AQI(1930)', '3HR_AQI(2230)']
                        ):
                            timestamp = base_date + timedelta(hours=hour_offset)
                            reshaped_data.append({
                                'time': timestamp,
                                'pm25_conc_cnn': row.get(conc_col, None),
                                'pm25_aqi': row.get(aqi_col, None),
                                'Station': row.get('Station', None),
                                'Site_Name': row.get('Site_Name', None)
                            })
                    except Exception as row_err:
                        if not silent:
                            print(f"Row error: {row_err}")
                        continue

                reshaped_df = pd.DataFrame(reshaped_data)
                all_data = pd.concat([all_data, reshaped_df], ignore_index=True)

            except Exception as e:
                if not silent:
                    print(f"Fetch failed {date.strftime('%Y-%m-%d')}: {e}")
                    continue

    # Empty data fallback
    if all_data.empty:
        if not silent:
            print("No MERRA2 data, filling nulls")
        all_data = pd.DataFrame({
            'time': pd.date_range(start=start_date, end=end_date, freq='3H'),
            'pm25_conc_cnn': [np.nan] * len(pd.date_range(start=start_date, end=end_date, freq='3H')),
            'pm25_aqi': [np.nan] * len(pd.date_range(start=start_date, end=end_date, freq='3H')),
            'Station': [None] * len(pd.date_range(start=start_date, end=end_date, freq='3H')),
            'Site_Name': [None] * len(pd.date_range(start=start_date, end=end_date, freq='3H'))
        })

    if not silent:
        print("Requesting GEOS-CF")

    geos_cf = mlpred.read_geos_cf(
        lon=lon,
        lat=lat,
        start=datetime.today() - timedelta(days=frequency),
        end=datetime.today() + timedelta(days=10),
        version=2
    )

    # Merge
    merg = funcs.merge_dataframes([all_data, geos_cf], "time", resample="3h", how="outer")
    print(merg.columns.to_list())

    # Fill missing values
    if not silent:
        print("Filling missing MERRA-2 values")
    
    filled_counts = {}
    
    # PM25 fill
    if 'pm25_conc_cnn' in merg.columns and 'pm25_rh35' in merg.columns:
        missing_mask = merg['pm25_conc_cnn'].isna()
        n_missing = missing_mask.sum()
        if n_missing > 0:
            merg.loc[missing_mask, 'pm25_conc_cnn'] = merg.loc[missing_mask, 'pm25_rh35']
            filled_counts['pm25_conc_cnn'] = n_missing
            if not silent:
                print(f"Filled {n_missing} pm25_conc_cnn with pm25_rh35")

    # NowCast calculation
    species_map = {'PM2.5': 'pm25_rh35', 'NO2': 'no2', 'O3': 'o3'}
    avg_hours = {'NO2': 3, 'O3': 1}
    merg = funcs.calculate_nowcast(merg, species_columns=species_map, avg_hours=avg_hours)

    # AQI fill
    if 'pm25_aqi' in merg.columns and 'PM25_NowCast_AQI' in merg.columns:
        missing_mask = merg['pm25_aqi'].isna()
        n_missing = missing_mask.sum()
        if n_missing > 0:
            merg.loc[missing_mask, 'pm25_aqi'] = merg.loc[missing_mask, 'PM25_NowCast_AQI']
            filled_counts['pm25_aqi'] = n_missing
            if not silent:
                print(f"Filled {n_missing} pm25_aqi with PM25_NowCast_AQI")
    
    # Summary
    if not silent and filled_counts:
        print(f"Filled: {filled_counts}")
    elif not silent:
        print("No fill needed")

    if not silent:
        print("Columns:", merg.columns.to_list())

    # Time conversion
    all_data = funcs.convert_times_column(merg, 'time', lat, lon)

    return all_data
