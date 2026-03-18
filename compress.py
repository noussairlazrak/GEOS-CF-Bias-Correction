#!/usr/bin/env python3
"""
Generate pre-computed hourly forecast snapshot files for 3 days in advance.
This script reads all individual site forecast files and creates lightweight
snapshots for every hour of the current day + 3 days ahead (96 hours total).

Usage:
    python3 compress.py
    
Output:
    precomputed/hourly_forecasts/YYYY-MM-DD_HH.json
    Example: precomputed/hourly_forecasts/2026-03-16_04.json (for 4:00 AM)
    
    Generates 96 files for current + 3 days ahead, rotating every hour.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import glob
import boto3

# Import GeoTIFF generation routines
from tiffs import (
    latest_init_date_and_hour,
    DEFAULT_VARIABLES,
    run as generate_tiffs,
)

# Import funcs for S3 operations
sys.path.insert(1, 'MLpred')
from MLpred import funcs

# S3 configuration
S3_BUCKET = "smce-geos-cf-forecasts-oss-shared"
S3_PREFIXES = [
    "snwg_forecast_working_files/precomputed/hourly_forecasts/",
    "snwg_forecast_working_files/precomputed/pmtiles_output/"
]

def get_forecast_for_hour(site_data, site_name, target_utc, site_index_entry=None):
    """Get forecast for a specific hour from site data"""
    try:
        timezone_str = site_data.get('timezone', 'UTC')
        target_hour_utc = target_utc.strftime('%Y-%m-%d %H:00:00')
        
        species = site_index_entry.get('species', 'no2') if site_index_entry else site_data.get('species', 'no2')
        sources = site_index_entry.get('sources', []) if site_index_entry else site_data.get('sources', [])
        
        for forecast in site_data.get('forecasts', []):
            if forecast.get('time') == target_hour_utc:
                return {
                    'location_name': site_name,
                    'timezone': timezone_str,
                    'species': species,
                    'sources': sources,
                    'observation_source': site_data.get('observation_source', 'NASA'),
                    'time': target_hour_utc,
                    'local_time': forecast.get('local_time'),
                    'no2': forecast.get('no2'),
                    'no2_aqi': forecast.get('no2_aqi'),
                    'o3': forecast.get('o3'),
                    'o3_aqi': forecast.get('o3_aqi'),
                    'pm25': forecast.get('pm25'),
                    'pm25_aqi': forecast.get('pm25_aqi'),
                    't10m': forecast.get('t10m'),
                    'rh': forecast.get('rh'),
                    'wind_speed': forecast.get('wind_speed')
                }
        
        closest_forecast = None
        min_diff = float('inf')
        target_hour = target_utc.replace(minute=0, second=0, microsecond=0)
        
        for forecast in site_data.get('forecasts', []):
            try:
                fc_time_str = forecast.get('time', '').replace(' ', 'T')
                fc_time = datetime.fromisoformat(fc_time_str)
                if fc_time.tzinfo is None:
                    fc_time = fc_time.replace(tzinfo=ZoneInfo('UTC'))
                
                diff = abs((fc_time - target_hour).total_seconds())
                if diff < min_diff:
                    min_diff = diff
                    closest_forecast = forecast
            except:
                pass
        
        if closest_forecast:
            return {
                'location_name': site_name,
                'timezone': timezone_str,
                'species': species,
                'sources': sources,
                'observation_source': site_data.get('observation_source', 'NASA'),
                'time': target_hour_utc,
                'local_time': closest_forecast.get('local_time'),
                'no2': closest_forecast.get('no2'),
                'no2_aqi': closest_forecast.get('no2_aqi'),
                'o3': closest_forecast.get('o3'),
                'o3_aqi': closest_forecast.get('o3_aqi'),
                'pm25': closest_forecast.get('pm25'),
                'pm25_aqi': closest_forecast.get('pm25_aqi'),
                't10m': closest_forecast.get('t10m'),
                'rh': closest_forecast.get('rh'),
                'wind_speed': closest_forecast.get('wind_speed')
            }
    except Exception as e:
        pass
    
    return None

def generate_all_hourly_snapshots():
    """Generate hourly snapshot files for 96 hours (current + 3 days)"""
    
    output_dir = Path('precomputed/hourly_forecasts')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    now_utc = datetime.now(ZoneInfo('UTC'))
    num_hours = 96
    
    print(f"Generating {num_hours} hourly forecast files...")
    print(f"Starting from: {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    sites_index_list = []
    try:
        if os.path.exists('precomputed/sites_index.json'):
            with open('precomputed/sites_index.json', 'r') as f:
                sites_index_list = json.load(f)
            print(f"Loaded {len(sites_index_list)} site entries from sites_index.json\n")
        else:
            print("Warning: sites_index.json not found, proceeding without it.")
    except Exception as e:
        print(f"ERROR: Failed to load sites_index.json: {e}")
    
    site_files = glob.glob('precomputed/all_dts/*.json')
    sites_data = {}
    
    print(f"Loading {len(site_files)} site files...")
    for site_file in site_files:
        try:
            site_name = Path(site_file).stem
            with open(site_file, 'r') as f:
                text = f.read()
                sanitized = text.replace('NaN', 'null')
                sites_data[site_name] = json.loads(sanitized)
        except Exception as e:
            print(f"Error loading {site_file}: {e}")
    
    print(f"Loaded {len(sites_data)} sites\n")
    
    generated_files = []
    failed_hours = 0
    
    for hour_offset in range(num_hours):
        try:
            target_utc = now_utc + timedelta(hours=hour_offset)
            filename = target_utc.strftime('%Y-%m-%d_%H.json')
            filepath = output_dir / filename
            
            hourly_data = {
                'generated_at': datetime.now(ZoneInfo('UTC')).isoformat(),
                'forecast_hour': target_utc.isoformat(),
                'sites': []
            }
            
            for site_name, site_data in sites_data.items():
                try:
                    site_index_entry = None
                    site_data_sources = site_data.get('sources', [])
                    
                    for index_entry in sites_index_list:
                        if index_entry.get('location_name') == site_name:
                            index_sources = set(index_entry.get('sources', []))
                            data_sources = set(site_data_sources)
                            
                            if index_sources & data_sources:
                                site_index_entry = index_entry
                                break
                    
                    forecast = get_forecast_for_hour(site_data, site_name, target_utc, site_index_entry)
                    if forecast:
                        hourly_data['sites'].append(forecast)
                except Exception as e:
                    pass
            
            if hourly_data['sites']:
                with open(filepath, 'w') as f:
                    json.dump(hourly_data, f, separators=(',', ':'))
                
                file_size_kb = filepath.stat().st_size / 1024
                generated_files.append(filename)
                
                # Upload to S3
                try:
                    s3_client = boto3.client("s3")
                    if not funcs.upload_to_s3(str(filepath), s3_client, S3_BUCKET):
                        print(f"Warning: Upload may have failed for {filename}")
                except Exception as e:
                    print(f"Warning: Could not upload {filename} to S3: {e}")
                
                if (hour_offset + 1) % 24 == 0:
                    print(f"  [{hour_offset + 1}/{num_hours}] {filename}: {file_size_kb:.1f}KB ({len(hourly_data['sites'])} sites)")
            else:
                failed_hours += 1
                
        except Exception as e:
            failed_hours += 1
    
    print(f"\nCleaning up old files (keeping last 96 hours)...")
    cutoff_time = datetime.now().timestamp() - (96 * 3600)
    removed_count = 0
    
    for old_file in output_dir.glob('*.json'):
        if old_file.stat().st_mtime < cutoff_time:
            old_file.unlink()
            removed_count += 1
    
    print(f"Removed {removed_count} old files")
    
    print(f"\n{'='*60}")
    print(f"Generated: {len(generated_files)} hourly forecast files")
    print(f"Directory: {output_dir}")
    print(f"Coverage: {num_hours} hours ({num_hours/24:.1f} days)")
    print(f"Time range:")
    print(f"    Start: {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"    End:   {(now_utc + timedelta(hours=num_hours-1)).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*60}")
    
    return len(generated_files)

if __name__ == '__main__':
    print("Generating hourly forecast snapshots for 3 days...\n")
    
    # Check S3 connectivity first
    print("=== Checking S3 Connectivity ===")
    try:
        s3_client = boto3.client("s3")
        funcs.check_s3_connectivity(S3_BUCKET, S3_PREFIXES)
        print("S3 bucket accessible\n")
    except Exception as e:
        print(f"Warning: S3 connectivity check failed: {e}\n")
    
    count = generate_all_hourly_snapshots()
    print(f"\nTotal hourly files: {count}")

    print("\n=== Generating GeoTIFF files ===")
    tiff_output_dir = Path('precomputed/pmtiles_output')
    try:
        init_date, init_hour = latest_init_date_and_hour()
        print(f"Latest GEOS-CF run: {init_date} {init_hour}z")
        
        generate_tiffs(
            init_date=init_date,
            init_hour=init_hour,
            variables=DEFAULT_VARIABLES,
            output_dir=tiff_output_dir,
            keep_nc4=False,
            days=5,
            tmp_dir_override=None,
        )
        print("GeoTIFF generation complete")
        
        # Upload to S3
        print("\n=== Uploading to S3 ===")
        tiff_files = list(tiff_output_dir.glob('*.tif')) + list(tiff_output_dir.glob('*.tiff'))
        
        if tiff_files:
            s3_client = boto3.client("s3")
            uploaded_count = 0
            
            for tiff_file in tiff_files:
                try:
                    if funcs.upload_to_s3(str(tiff_file), s3_client, S3_BUCKET):
                        uploaded_count += 1
                except Exception as e:
                    print(f"Warning: Failed to upload {tiff_file.name}: {e}")
            
            if uploaded_count == len(tiff_files):
                print(f"Successfully uploaded all {uploaded_count} GeoTIFF files")
            else:
                print(f"Uploaded {uploaded_count}/{len(tiff_files)} GeoTIFF files")
        else:
            print("No GeoTIFF files to upload")
            
    except Exception as e:
        print(f"Warning: GeoTIFF generation failed: {e}", file=sys.stderr)
        print("Hourly forecasts still saved successfully.")

