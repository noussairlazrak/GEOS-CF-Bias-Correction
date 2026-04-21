import sys
from IPython.display import HTML
sys.path.insert(1,'MLpred')

import os
from MLpred import mlpred
from MLpred import funcs
from MLpred.s3_manager import S3Manager
from MLpred.geos_fp_cnn import read_geos_fp_cnn
import datetime as dt
import pandas as pd
import numpy as np
import requests
import json
import boto3
import traceback
import argparse
import joblib 

# Arguments
parser = argparse.ArgumentParser(description='Generate and upload forecasts to S3')
parser.add_argument('--skip-plotting', action='store_true', default=False, help='Skip plot generation')
parser.add_argument('--skip-openaq', action='store_true', default=False, help='Skip OpenAQ data retrieval')
parser.add_argument('--s3-only', action='store_true', default=False, help='Only upload to S3 (keeps local copies)')
parser.add_argument('--clean-local', action='store_true', default=False, help='Remove local files after S3 upload')
parser.add_argument('--no-local-save', action='store_true', default=False, help='Save directly to S3 without local files')
parser.add_argument('--model-cache', choices=['local', 's3'], default='s3', help='Store model CSV cache locally or on S3 (default: s3)')
parser.add_argument('--force-update', action='store_true', default=False, help='Force update even if files are recent')
parser.add_argument('--stale-hours', type=int, default=48, help='Hours threshold for considering files stale (default: 48)')
args = parser.parse_args()

# Config
SKIP_PLOTTING = args.skip_plotting
SKIP_OPENAQ = args.skip_openaq
S3_ONLY = args.s3_only
CLEAN_LOCAL = args.clean_local
NO_LOCAL_SAVE = args.no_local_save
SAVE_PLOTS_LOCAL = True
UPLOAD_PLOTS_S3 = True
FORECAST_HOURS_THRESHOLD = 5
STALE_HOURS = args.stale_hours
FORCE_UPDATE = args.force_update

# Cache
MODEL_CACHE_SOURCE = args.model_cache

# S3
S3_BUCKET = "smce-geos-cf-public"
S3_PREFIXES = {
    "geos_cf": "snwg_forecast_working_files/GEOS_CF",
    "openaq": "snwg_forecast_working_files/OPENAQ",
    "plots": "snwg_forecast_working_files/plots",
    "forecasts": "snwg_forecast_working_files/precomputed/all_dts",
    "models": "snwg_forecast_working_files/MODELS",
    "geos_fp_cnn": "snwg_forecast_working_files/GEOS_FP_CNN"
}

# Init
s3_manager = S3Manager(bucket_name=S3_BUCKET)
s3_client = boto3.client("s3")

# Legacy
s3_bucket = S3_BUCKET
s3_prefixes = list(S3_PREFIXES.values())

# Connectivity
print("Checking S3 connectivity...")
connectivity = s3_manager.check_connectivity(s3_prefixes)
for prefix, status in connectivity.items():
    print(f"  {prefix}: {'OK' if status else 'FAIL'}")

# Directories
if not NO_LOCAL_SAVE:
    local_dirs = ["./precomputed/all_dts/", "./plots/"]
    for local_dir in local_dirs:
        os.makedirs(local_dir, exist_ok=True)
        print(f"Directory ready: {local_dir}")
else:
    print(f"NO_LOCAL_SAVE mode enabled")

if S3_ONLY:
    print(f"S3_ONLY mode enabled")
if CLEAN_LOCAL:
    print(f"CLEAN_LOCAL mode enabled")

# Locations
url = "https://raw.githubusercontent.com/noussairlazrak/MLpred/refs/heads/main/global.json"
print(f"Config: SKIP_PLOTTING={SKIP_PLOTTING}, SKIP_OPENAQ={SKIP_OPENAQ}, S3_ONLY={S3_ONLY}, CLEAN_LOCAL={CLEAN_LOCAL}, NO_LOCAL_SAVE={NO_LOCAL_SAVE}, MODEL_CACHE_SOURCE={MODEL_CACHE_SOURCE}, STALE_HOURS={STALE_HOURS}")
data = json.loads(requests.get(url, stream=True).text)
all_locations = [] 

# Forecasts
for key, location_data in list(data.items()):
    if location_data.get("observation_source") in ("DoS_Missions", "NASA Pandora", "REMMAQ"):
        site = location_data['location_name'].replace(" ", "_")
        locname = location_data["location_name"]
        lat = location_data["lat"]
        lon = location_data["lon"]
        print(f"\nProcessing: {locname} (lat: {lat}, lon: {lon})")
        
        if location_data["observation_source"] == "DoS_Missions":
            # Paths
            site_file_path = f'./precomputed/all_dts/{site}.json'
            s3_key = f"{S3_PREFIXES['forecasts']}/{site}.json"
            
            # Freshness
            if not FORCE_UPDATE:
                if NO_LOCAL_SAVE:
                    if s3_manager.file_exists(s3_key) and s3_manager.is_file_recent(s3_key, hours_threshold=STALE_HOURS):
                        age = s3_manager.get_file_age_hours(s3_key)
                        print(f"Skipping {locname} - recent ({age:.1f}h old)")
                        continue
                else:
                    if funcs.is_forecast_recent(site_file_path, hours_threshold=FORECAST_HOURS_THRESHOLD):
                        print(f"Skipping {locname} - recent")
                        continue
                
            # Settings
            site_settings = {'l_name': locname, 
             'species': 'no2', 
             'lat': lat, 
             'lon': lon,
             'silent': True,
             'model_src': 's3',
            }

            try:
                merra2cnn = read_geos_fp_cnn(site=site, frequency=5, lat=lat, lon=lon, silent=False, skip_geosfp=True)
                metadata = None
                
                # Forecast
                forecast_dict = {
                    "location": site_settings.get("l_name", "N/A"),
                    "lat": site_settings.get("lat", "N/A"),
                    "lon": site_settings.get("lon", "N/A"),
                    "species": "pm25",
                    "sources": ["merra2", "geoscf"],
                    "forecasts": merra2cnn if isinstance(merra2cnn, list) else merra2cnn.to_dict(orient='records') if hasattr(merra2cnn, 'to_dict') else merra2cnn,
                    "metadata": metadata
                }
                
                # Save
                if NO_LOCAL_SAVE:
                    if s3_manager.upload_json(forecast_dict, s3_key):
                        print(f"PM2.5 forecast for {locname} saved to S3")
                    else:
                        print(f"Failed to save PM2.5 forecast for {locname}")
                else:
                    try:
                        with open(site_file_path, 'w') as f:
                            json.dump(forecast_dict, f, indent=2, default=str)
                        print(f"PM2.5 forecast for {locname} saved locally")
                        
                        # Upload
                        s3_manager.upload_file(site_file_path, s3_key)
                        print(f"PM2.5 forecast for {locname} uploaded to S3")
                        
                        # Cleanup
                        if CLEAN_LOCAL:
                            os.remove(site_file_path)
                            print(f"Local file removed: {site_file_path}")
                    except Exception as e:
                        print(f"Error saving PM2.5 forecast for {locname}: {e}")
                    
            except Exception as e:
                print(f"Error processing merra2 for {key}: {e}")
                traceback.print_exc()
            
        if location_data["observation_source"] in ("NASA Pandora", "REMMAQ"):
            # Paths
            file_path = f'./precomputed/all_dts/{locname}.json'
            s3_key = f"{S3_PREFIXES['forecasts']}/{locname}.json"
            
            # Freshness
            if not FORCE_UPDATE:
                if NO_LOCAL_SAVE:
                    if s3_manager.file_exists(s3_key) and s3_manager.is_file_recent(s3_key, hours_threshold=STALE_HOURS):
                        age = s3_manager.get_file_age_hours(s3_key)
                        print(f"Skipping {locname} - recent ({age:.1f}h old)")
                        continue
                else:
                    if funcs.is_forecast_recent(file_path, hours_threshold=FORECAST_HOURS_THRESHOLD):
                        print(f"Skipping {locname} - recent")
                        continue
            
            # Source
            source_type = location_data["observation_source"]
            if source_type == "NASA Pandora":
                obs_src = 'pandora'
                col_name = 'pandora'
                time_col = location_data["obs_options"]["no2"]["time_col"]
                date_format = '%Y-%m-%d %H:%M'
                obs_val_col = location_data["obs_options"]["no2"]["val_col"]
                obs_url = f'{location_data["obs_options"]["no2"]["file"]}'
                
                
            else:  # REMMAQ
                obs_src = 'local'
                col_name = 'local'
                time_col = location_data["obs_options"]["no2"]["time_col"]
                date_format = location_data["obs_options"]["no2"]["time_parser"]
                obs_val_col = location_data["obs_options"]["no2"]["val_col"]
                obs_url = f'REMMAQ/{location_data["obs_options"]["no2"]["file"]}'
            
            print(f"Source: {obs_src} for {locname}")
            try:
                site_settings = {'l_name': locname, 
                 'species': 'no2', 
                 'lat': lat, 
                 'lon': lon,
                 'silent': True,
                 'model_src': 's3',
                 'obs_src': obs_src,
                 'openaq_id': None,
                 'model_tuning' : False,
                 'model_url': '#',
                 'obs_url': obs_url,
                 'resample' : '1h',
                 'unit' : 'ppb',
                 'interpolation': True,
                 'remove_outlier': True,
                 'start' : dt.datetime(2018, 1, 1),   
                 'end': dt.datetime.today()
                }
                obs_settings = {'time_col': time_col, 
                                     'date_format': date_format, 
                                     'obs_val_col': obs_val_col, 
                                     'lat_col': None, 
                                     'lon_col': None,
                                     'remove_outlier': False,
                                    }
                merged_data, metrics, model = mlpred.get_localised_forecast(
                    loc=site_settings['l_name'],
                    spec=site_settings['species'],
                    lat=site_settings['lat'],
                    lon=site_settings['lon'],
                    mod_src=site_settings['model_src'],
                    obs_src=site_settings['obs_src'],
                    openaq_id=site_settings['openaq_id'],
                    GEOS_CF=site_settings['model_url'],
                    OBS_URL=site_settings['obs_url'],
                    st=site_settings['start'],
                    ed=site_settings['end'],
                    resamp=site_settings['resample'],
                    unit=site_settings['unit'],
                    interpol=site_settings['interpolation'],
                    rmv_out=site_settings['remove_outlier'],
                    time_col=obs_settings['time_col'],
                    date_fmt=obs_settings['date_format'],
                    obs_val_col=obs_settings['obs_val_col'],
                    lat_col=obs_settings['lat_col'],
                    lon_col=obs_settings['lon_col'],
                    silent=site_settings['silent'],
                    force_retrain=True
                )
                if merged_data is None:
                    print(f"ERROR: Bias correction failed for {locname}")
                    continue
                
                # Process
                forecasts_raw = merged_data
                metadata = metrics
                col_map = {"time": "time", "no2": "no2", "localised": "corrected", "value": col_name}
                fcast = forecasts_raw[[c for c in ["time", "no2", "localised", "value", "o3", "pm25_rh35", "rh","t","tprec","hcho"] if c in forecasts_raw.columns]].rename(columns=col_map)
                fcast.iloc[:, 1:] = fcast.iloc[:, 1:].clip(lower=0)

                start, end = fcast["time"].min(), fcast["time"].max()
                hourly_index = pd.date_range(start, end, freq="1H")
                aq_df = pd.DataFrame({"time": hourly_index})

                if not SKIP_OPENAQ:
                    sensors = [
                        s["id"]
                        for loc in mlpred.get_openaq_locations(lat=float(lat), lon=float(lon), radius=25, parameter="no2")
                        for s in loc.get("sensors", [])
                    ]
                    print(sensors)
                    if sensors:
                        aq_data = funcs.openaq_hourly_avgs(sensors, start, end)
                        if not aq_data.empty:
                            aq_df = pd.merge(aq_df, aq_data, on="time", how="outer")
                else:
                    print(f"Skipping OpenAQ")

                fcast = fcast.set_index("time").reindex(hourly_index).reset_index().rename(columns={"index": "time"})
                merg = funcs.merge_dataframes([aq_df, fcast], "time", resample="1h", how="outer")

                for col in ["no2", "corrected", col_name, "avg"]:
                    merg[col] = merg.get(col, np.nan)

                merg["corrected"] = merg["corrected"].apply(lambda x: x if x > 0 else 0.1)
                merg["avg"] = merg["avg"]*1000
                merg[merg.select_dtypes(include=["float", "int"]).columns] = merg.select_dtypes(include=["float", "int"]).round(2)

                merg_plot = merg.set_index("time").resample("5D").mean()
                merg_plot = merg_plot.reset_index()
                
                # Plotting
                if not SKIP_PLOTTING:
                    try:
                        local_plot_dir = "./plots"
                        os.makedirs(local_plot_dir, exist_ok=True)
                        local_plot_path = os.path.join(local_plot_dir, f"{locname}.png")
                        
                        funcs.gen_plot(
                            merg_plot,
                            [['no2', 'corrected', col_name, 'avg']],
                            [['black', 'grey', 'red', 'green']],
                            [['--', '-', '-', '-']],
                            'NO2',
                            [f'{locname}'],
                            sv_pth=local_plot_path,
                            resample='1D'
                        )
                        print(f"Plot generated for {locname}")
                        
                        # Upload
                        if UPLOAD_PLOTS_S3:
                            plot_s3_key = f"{S3_PREFIXES['plots']}/{locname}.png"
                            if s3_manager.upload_file(local_plot_path, plot_s3_key):
                                print(f"Plot for {locname} uploaded to S3")
                                if CLEAN_LOCAL and os.path.exists(local_plot_path):
                                    os.remove(local_plot_path)
                                    print(f"Local plot removed: {local_plot_path}")
                        
                    except Exception as plot_err:
                        print(f"Plot failed for {locname}: {plot_err}")
                        traceback.print_exc()

                # Cutoff
                cutoff = fcast["time"].max() - pd.DateOffset(months=12)
                merg = merg[merg["time"] >= cutoff]
                merg = funcs.convert_times_column(merg, 'time', lat, lon)
                
                # NowCast
                species_map = { 'PM2.5': 'pm25_rh35', 'NO2': 'corrected', 'O3': 'o3' } 
                avg_hours = { 'NO2': 3, 'O3': 1 }

                merg = funcs.calculate_nowcast(merg, species_columns=species_map, avg_hours=avg_hours)
                print(merg.columns)

                # Forecast
                forecast_dict = {
                    "location": site_settings.get("l_name", "N/A"),
                    "lat": site_settings.get("lat", "N/A"),
                    "lon": site_settings.get("lon", "N/A"),
                    "species": "no2",
                    "sources": ["geoscf", col_name],
                    "forecasts": merg.to_dict(orient='records'),
                    "metrics": metadata,
                    "metadata": metadata
                }
                
                # Save
                if NO_LOCAL_SAVE:
                    if s3_manager.upload_json(forecast_dict, s3_key):
                        print(f"NO2 forecast for {locname} saved to S3")
                    else:
                        print(f"Failed to save NO2 forecast for {locname}")
                else:
                    try:
                        with open(file_path, 'w') as f:
                            json.dump(forecast_dict, f, indent=2, default=str)
                        print(f"NO2 forecast for {locname} saved locally")
                        
                        # Upload
                        s3_manager.upload_file(file_path, s3_key)
                        print(f"NO2 forecast for {locname} uploaded to S3")
                        
                        # Cleanup
                        if CLEAN_LOCAL:
                            os.remove(file_path)
                            print(f"Local file removed: {file_path}")
                    except Exception as e:
                        print(f"Error saving NO2 forecast for {locname}: {e}")

            except Exception as e:
                print(f"Error processing {source_type} for {key}: {e}")
                traceback.print_exc()

print("\nForecast generation completed.")
