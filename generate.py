import sys
from IPython.display import HTML
sys.path.insert(1,'MLpred')

import os
from MLpred import mlpred
from MLpred import funcs
from MLpred.bias_corrector import get_localised_forecast
from MLpred.s3_manager import S3Manager
from MLpred.geos_fp_cnn import read_geos_fp_cnn, load_geojson_all_locations
from MLpred.alerts import alert_read_failure, alert_no_new_data
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
parser.add_argument('--force-source', nargs='+', default=[], metavar='SOURCE',
                    help='Force update only for specific observation sources (e.g. --force-source "NASA Pandora" DoS_Missions)')
parser.add_argument('--skip-source', nargs='+', default=[], metavar='SOURCE',
                    help='Skip one or more observation sources entirely (e.g. --skip-source DoS_Missions REMMAQ)')
parser.add_argument('--stale-hours', type=int, default=48, help='Hours threshold for considering files stale (default: 48)')
parser.add_argument('--locations', type=str, default=None, metavar='LOCATIONS',
                    help='Run forecasts for specific locations. Can be: comma-separated names (e.g. "Agam,Beijing-RADI,Tokyo-Sophia") or path to file with location names (one per line)')
parser.add_argument('--location-id', type=str, default=None, metavar='ID',
                    help='Run forecast for a single location by its global.json id (e.g. 184 or -184)')
parser.add_argument('--s3-anon', action='store_true', default=False,
                    help='Use anonymous (unsigned) S3 access — for local testing without AWS credentials')
parser.add_argument('--plot-full-history', action='store_true', default=False,
                    help='Also plot corrected vs. estimated NO2 over the full GEOS-CF period')
parser.add_argument('--full-history-resample', type=str, default='7D', metavar='FREQ',
                    help='Resample frequency for the full-history plot (default: 7D)')
args = parser.parse_args()

# Config
SKIP_PLOTTING = args.skip_plotting
SKIP_OPENAQ = args.skip_openaq
S3_ONLY = args.s3_only
CLEAN_LOCAL = args.clean_local
NO_LOCAL_SAVE = args.no_local_save
SAVE_PLOTS_LOCAL = True
UPLOAD_PLOTS_S3 = True
PLOT_FULL_HISTORY = args.plot_full_history
FULL_HISTORY_RESAMPLE = args.full_history_resample
FORECAST_HOURS_THRESHOLD = 3
STALE_HOURS = args.stale_hours
FORCE_UPDATE = args.force_update
FORCE_SOURCES = [s.strip() for s in args.force_source]  
SKIP_SOURCES  = [s.strip() for s in args.skip_source]   

# Parse location filter
LOCATION_FILTER = None
if args.locations:
    if os.path.isfile(args.locations):
        # Read from file
        with open(args.locations, 'r') as f:
            LOCATION_FILTER = set(line.strip() for line in f if line.strip())
        print(f"Location filter loaded from file: {len(LOCATION_FILTER)} locations")
    else:
        # Parse comma-separated list
        LOCATION_FILTER = set(loc.strip() for loc in args.locations.split(',') if loc.strip())
        print(f"Location filter from argument: {len(LOCATION_FILTER)} locations")

LOCATION_ID_FILTER = args.location_id.lstrip('-') if args.location_id else None
if LOCATION_ID_FILTER:
    print(f"Location ID filter: {LOCATION_ID_FILTER}")

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
s3_manager = S3Manager(bucket_name=S3_BUCKET, anon=args.s3_anon)
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
config_msg = f"Config: SKIP_PLOTTING={SKIP_PLOTTING}, SKIP_OPENAQ={SKIP_OPENAQ}, S3_ONLY={S3_ONLY}, CLEAN_LOCAL={CLEAN_LOCAL}, NO_LOCAL_SAVE={NO_LOCAL_SAVE}, MODEL_CACHE_SOURCE={MODEL_CACHE_SOURCE}, STALE_HOURS={STALE_HOURS}, FORCE_SOURCES={FORCE_SOURCES}, SKIP_SOURCES={SKIP_SOURCES}, PLOT_FULL_HISTORY={PLOT_FULL_HISTORY}"
if LOCATION_FILTER:
    config_msg += f", LOCATION_FILTER={len(LOCATION_FILTER)} locations"
print(config_msg)
data = json.loads(requests.get(url, stream=True).text)
all_locations = [] 

# Pre-check
print("\nChecking GEOS-FP endpoint connectivity …")
_today = dt.datetime.today()
SKIP_GEOSFP = True
for _day_offset in (0, 1):
    _attempt = _today - dt.timedelta(days=_day_offset)
    _label = "current" if _day_offset == 0 else "-1 day"
    print(f"  Trying {_label} ({_attempt.strftime('%Y-%m-%d')}) …")
    _gdf = load_geojson_all_locations(date=_attempt, silent=False)
    if not _gdf.empty:
        print(f"  GEOS-FP GeoJSON available ({_label}).")
        SKIP_GEOSFP = False
        break
else:
    print("No GEOS-FP GeoJSON found")
    alert_no_new_data(
        "load_geojson_all_locations", "all DoS_Missions locations",
        reason="No GEOS-FP GeoJSON available for current day or -1 day; "
               "PM2.5 forecasts will be skipped this run.",
    )

# Forecasts
for key, location_data in list(data.items()):
    if LOCATION_ID_FILTER and key.lstrip('-') != LOCATION_ID_FILTER:
        continue
    if location_data.get("observation_source") in ("DoS_Missions", "NASA Pandora", "REMMAQ"):
        obs_source = location_data["observation_source"]

        if SKIP_SOURCES and obs_source in SKIP_SOURCES:
            print(f"Skipping {obs_source} (--skip-source)")
            continue

        site = location_data['location_name'].replace(" ", "_")
        locname = location_data["location_name"]
        lat = location_data["lat"]
        lon = location_data["lon"]
        
        # Apply location filter
        if LOCATION_FILTER and locname not in LOCATION_FILTER:
            continue
        
        print(f"\nProcessing: {locname} (lat: {lat}, lon: {lon})")

        force_this = FORCE_UPDATE or (obs_source in FORCE_SOURCES)

        if location_data["observation_source"] == "DoS_Missions":
            # Paths
            site_file_path = f'./precomputed/all_dts/{site}.json'
            s3_key = f"{S3_PREFIXES['forecasts']}/{site}.json"
            
            # Freshness
            if not force_this:
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
             'silent': False,
             'model_src': 's3',
            }

            try:
                merra2cnn = read_geos_fp_cnn(site=site, frequency=5, lat=lat, lon=lon, silent=False, skip_geosfp=SKIP_GEOSFP)
                merra2cnn = funcs.calculate_overall_aqi(merra2cnn)
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
                alert_read_failure("read_geos_fp_cnn", locname, e, lat=lat, lon=lon)
            
        if location_data["observation_source"] in ("NASA Pandora", "REMMAQ"):
            # Paths
            file_path = f'./precomputed/all_dts/{locname}.json'
            s3_key = f"{S3_PREFIXES['forecasts']}/{locname}.json"
            
            # Freshness
            if not force_this:
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
                 'silent': False,
                 'model_src': 's3',
                 'obs_src': obs_src,
                 'openaq_id': None,
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
                merged_data, metrics, model = get_localised_forecast(
                    loc=site_settings['l_name'],
                    spec=site_settings['species'],
                    lat=site_settings['lat'],
                    lon=site_settings['lon'],
                    mod_src=site_settings['model_src'],
                    obs_src=site_settings['obs_src'],
                    openaq_id=site_settings['openaq_id'],
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
                    force_retrain=False,    
                    force_obs_refresh=False,
                    model_max_age_days=7,  
                    s3_manager=s3_manager,
                )
                if merged_data is None:
                    print(f"ERROR: Bias correction failed for {locname}")
                    alert_no_new_data(
                        "get_localised_forecast", locname,
                        reason="read_pandora/read_geos_cf/read_obs returned no usable data "
                               "(see log above for the specific ERROR line)",
                        obs_source=obs_src, lat=lat, lon=lon,
                    )
                    continue
                

                metadata = metrics
                if metrics:
                    perf_bits = [f"{k}={metrics[k]}" for k in
                                 ("source", "target", "R2", "RMSE", "MAE",
                                  "CV_R2", "CV_RMSE", "CV_MAE", "Train_R2", "n_train")
                                 if metrics.get(k) is not None]
                    print(f"Model performance for {locname}: "
                          f"{', '.join(perf_bits) if perf_bits else 'no metrics available'}")
                col_map = {"no2": "no2", "localised": "corrected", "value": col_name}
                keep_cols = [c for c in ["time", "no2", "localised", "value", "o3", "pm25_rh35", "rh", "t", "tprec", "hcho"] if c in merged_data.columns]
                fcast = merged_data[keep_cols].rename(columns=col_map).copy()

                # Deduplicate and sort
                fcast = fcast.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

                # Clip negatives
                num_cols = fcast.select_dtypes(include=["float", "int"]).columns
                fcast[num_cols] = fcast[num_cols].clip(lower=0)
                
                mask = fcast['corrected'] > fcast[col_name].max()
                fcast.loc[mask, 'corrected'] = fcast.loc[mask, 'no2']

                # corrected is at least 0.1
                if "corrected" in fcast.columns:
                    fcast["corrected"] = fcast["corrected"].apply(lambda x: x if pd.notna(x) and x > 0 else 0.1)

                # Round numerics
                fcast[fcast.select_dtypes(include=["float", "int"]).columns] = fcast.select_dtypes(include=["float", "int"]).round(2)

                # Plotting
                if not SKIP_PLOTTING:
                    try:
                        local_plot_dir = "./plots"
                        os.makedirs(local_plot_dir, exist_ok=True)
                        local_plot_path = os.path.join(local_plot_dir, f"{locname}.png")
                        plot_cols = [c for c in ["no2", "corrected", col_name] if c in fcast.columns]
                        colors = ['black', 'grey', 'red'][:len(plot_cols)]
                        styles = ['--', '-', '-'][:len(plot_cols)]
                        fcast_plot = fcast.set_index("time").resample("5D").mean().reset_index()
                        funcs.gen_plot(
                            fcast_plot,
                            [plot_cols],
                            [colors],
                            [styles],
                            'NO2',
                            [f'{locname}'],
                            sv_pth=local_plot_path,
                            resample='1D'
                        )
                        print(f"Plot generated for {locname}")
                        if UPLOAD_PLOTS_S3:
                            plot_s3_key = f"{S3_PREFIXES['plots']}/{locname}.png"
                            if s3_manager.upload_file(local_plot_path, plot_s3_key):
                                print(f"Plot for {locname} uploaded to S3")
                                if CLEAN_LOCAL and os.path.exists(local_plot_path):
                                    os.remove(local_plot_path)
                    except Exception as plot_err:
                        print(f"Plot failed for {locname}: {plot_err}")
                        traceback.print_exc()

                    if PLOT_FULL_HISTORY:
                        try:
                            local_hist_plot_path = os.path.join(local_plot_dir, f"{locname}_historical.png")
                            hist_cols = [c for c in ["no2", "corrected"] if c in fcast.columns]
                            hist_colors = ['black', 'red'][:len(hist_cols)]
                            hist_styles = ['--', '-'][:len(hist_cols)]
                            funcs.gen_plot(
                                fcast,
                                [hist_cols],
                                [hist_colors],
                                [hist_styles],
                                'NO2',
                                [f'{locname} — full GEOS-CF period'],
                                sv_pth=local_hist_plot_path,
                                lbl_rnm={"no2": "Estimated (GEOS-CF)", "corrected": "Corrected"},
                                resample=FULL_HISTORY_RESAMPLE,
                            )
                            print(f"Full-history plot generated for {locname}")
                            if UPLOAD_PLOTS_S3:
                                hist_plot_s3_key = f"{S3_PREFIXES['plots']}/{locname}_historical.png"
                                if s3_manager.upload_file(local_hist_plot_path, hist_plot_s3_key):
                                    print(f"Full-history plot for {locname} uploaded to S3")
                                    if CLEAN_LOCAL and os.path.exists(local_hist_plot_path):
                                        os.remove(local_hist_plot_path)
                        except Exception as plot_err:
                            print(f"Full-history plot failed for {locname}: {plot_err}")
                            traceback.print_exc()


                # Two precomputed variants per location:
                #   - "<locname>.json": last 1 month + forecast tail (the default/app view)
                #   - "<locname>_historical.json": the full merged series (2018 -> forecast tail)
                historical_file_path = f'./precomputed/all_dts/{locname}_historical.json'
                historical_s3_key = f"{S3_PREFIXES['forecasts']}/{locname}_historical.json"
                cutoff_recent = pd.Timestamp(dt.datetime.now()) - pd.DateOffset(months=1)

                variants = [
                    (fcast[fcast["time"] >= cutoff_recent].copy(), file_path, s3_key, "recent (1mo + forecast)"),
                    (fcast.copy(), historical_file_path, historical_s3_key, "full historical"),
                ]

                for merg, out_path, out_s3_key, variant_label in variants:
                    merg = funcs.convert_times_column(merg, 'time', lat, lon)

                    # NowCast
                    species_map = {'PM2.5': 'pm25_rh35', 'NO2': 'corrected', 'O3': 'o3'}
                    avg_hours = {'NO2': 3, 'O3': 1}
                    merg = funcs.calculate_nowcast(merg, species_columns=species_map, avg_hours=avg_hours)
                    merg = funcs.calculate_overall_aqi(merg)

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
                        if s3_manager.upload_json(forecast_dict, out_s3_key):
                            print(f"NO2 forecast ({variant_label}) for {locname} saved to S3")
                        else:
                            print(f"Failed to save NO2 forecast ({variant_label}) for {locname}")
                    else:
                        try:
                            with open(out_path, 'w') as f:
                                json.dump(forecast_dict, f, indent=2, default=str)
                            print(f"NO2 forecast ({variant_label}) for {locname} saved locally")

                            # Upload
                            s3_manager.upload_file(out_path, out_s3_key)
                            print(f"NO2 forecast ({variant_label}) for {locname} uploaded to S3")

                            # Cleaning
                            if CLEAN_LOCAL:
                                os.remove(out_path)
                                print(f"Local file removed: {out_path}")
                        except Exception as e:
                            print(f"Error saving NO2 forecast ({variant_label}) for {locname}: {e}")

            except Exception as e:
                print(f"Error processing {source_type} for {key}: {e}")
                traceback.print_exc()
                alert_read_failure(
                    "get_localised_forecast", locname, e,
                    obs_source=obs_src, lat=lat, lon=lon,
                )

print("\nForecast generation completed.")