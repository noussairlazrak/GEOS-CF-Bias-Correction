#!/usr/bin/env python3
"""
Generate pre-computed hourly forecast snapshot files for 3 days in advance.
This script reads all individual site forecast files and creates lightweight
snapshots for every hour of the current day + 3 days ahead (96 hours total).

Usage:
    python3 generate_hourly_forecasts.py
    
Output:
    precomputed/hourly_forecasts/YYYY-MM-DD_HH.json
    Example: precomputed/hourly_forecasts/2025-11-19_14.json (for 2:00 PM)
    
    Generates 96 files for current + 3 days ahead, rotating every hour.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import glob

def get_forecast_for_hour(site_data, site_name, target_datetime):
    """Extract forecast for a specific hour from a site's data"""
    try:
        timezone_str = site_data.get('timezone', 'UTC')
        tz = ZoneInfo(timezone_str)
        
        target_local = target_datetime.astimezone(tz)
        hour_key = target_local.strftime('%Y-%m-%d %H:00:00')
        
        for forecast in site_data.get('forecasts', []):
            if forecast.get('local_time') == hour_key:
                return {
                    'location_name': site_name,
                    'timezone': timezone_str,
                    'species': site_data.get('species', 'no2'),
                    'observation_source': site_data.get('observation_source', 'NASA'),
                    'local_time': hour_key,
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
        
        try:
            target_hour = target_local.replace(minute=0, second=0, microsecond=0)
            future_found = False
            
            for forecast in site_data.get('forecasts', []):
                try:
                    fc_time_str = forecast.get('local_time', '').replace(' ', 'T')
                    fc_time = datetime.fromisoformat(fc_time_str)
                    if fc_time.tzinfo is None:
                        fc_time = fc_time.replace(tzinfo=tz)
                    
                    diff = (fc_time - target_hour).total_seconds()
                    
                    if diff >= 0 and diff < min_diff:
                        min_diff = diff
                        closest_forecast = forecast
                        future_found = True

                    elif not future_found and abs(diff) < min_diff:
                        min_diff = abs(diff)
                        closest_forecast = forecast
                except:
                    pass
        except:
            pass
        
        if closest_forecast:
            return {
                'location_name': site_name,
                'timezone': timezone_str,
                'species': site_data.get('species', 'no2'),
                'observation_source': site_data.get('observation_source', 'NASA'),
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
        print(f"Error processing {site_name} for hour: {e}")
    
    return None

def generate_all_hourly_snapshots():
    """Generate hourly snapshot files for current day + 3 days ahead (96 hours)"""
    

    output_dir = Path('precomputed/hourly_forecasts')
    output_dir.mkdir(parents=True, exist_ok=True)
    

    now_utc = datetime.now(ZoneInfo('UTC'))
    
    # Generate for 96 hours (current day + 3 days ahead)
    num_hours = 96
    
    print(f"Generating {num_hours} hourly forecast files...")
    print(f"Starting from: {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    

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
    
    # Generate snapshots for each hour
    generated_files = []
    failed_hours = 0
    
    for hour_offset in range(num_hours):
        try:
            # Calculate target hour in UTC
            target_utc = now_utc + timedelta(hours=hour_offset)
            filename = target_utc.strftime('%Y-%m-%d_%H.json')
            filepath = output_dir / filename
            
            # Collect all forecasts for this hour
            hourly_data = {
                'generated_at': datetime.now(ZoneInfo('UTC')).isoformat(),
                'forecast_hour': target_utc.isoformat(),
                'sites': []
            }
            
            # Process all sites
            for site_name, site_data in sites_data.items():
                try:
                    forecast = get_forecast_for_hour(site_data, site_name, target_utc)
                    if forecast:
                        hourly_data['sites'].append(forecast)
                except Exception as e:
                    pass  # Skip sites with errors for this hour
            
            # Write snapshot file
            if hourly_data['sites']:
                with open(filepath, 'w') as f:
                    json.dump(hourly_data, f, separators=(',', ':'))
                
                file_size_kb = filepath.stat().st_size / 1024
                generated_files.append(filename)
                
                if (hour_offset + 1) % 24 == 0:
                    print(f"  [{hour_offset + 1}/{num_hours}] Generated {filename}: {file_size_kb:.1f}KB ({len(hourly_data['sites'])} sites)")
            else:
                failed_hours += 1
                
        except Exception as e:
            print(f"Error generating snapshot for hour {hour_offset}: {e}")
            failed_hours += 1
    
    print(f"\nCleaning up old files (keeping last 96 hours)...")
    cutoff_time = datetime.now().timestamp() - (96 * 3600)
    removed_count = 0
    
    for old_file in output_dir.glob('*.json'):
        if old_file.stat().st_mtime < cutoff_time:
            old_file.unlink()
            removed_count += 1
    
    print(f"Removed {removed_count} old files")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✓ Generated: {len(generated_files)} hourly forecast files")
    print(f"✓ Failed: {failed_hours} hours")
    print(f"✓ Directory: {output_dir}")
    print(f"✓ Coverage: {num_hours} hours ({num_hours/24:.1f} days)")
    print(f"✓ Time range:")
    print(f"    Start: {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"    End:   {(now_utc + timedelta(hours=num_hours-1)).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*60}")
    
    return len(generated_files)

if __name__ == '__main__':
    print("Generating 3-day ahead hourly forecast snapshots...\n")
    count = generate_all_hourly_snapshots()
    print(f"\nDone! Total files: {count}")

