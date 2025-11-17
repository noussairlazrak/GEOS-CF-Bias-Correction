import sys
from IPython.display import HTML
sys.path.insert(1,'MLpred')

import os
from MLpred import mlpred
from MLpred import funcs
import datetime as dt
import pandas as pd
import numpy as np
import requests
import json
import boto3
import traceback

# S3 bucket and prefixes to check
s3_bucket = "localized-forecasts"
s3_prefixes = [
    "snwg_forecast_working_files/GEOS_CF/",
    "snwg_forecast_working_files/OPENAQ/",
    "snwg_forecast_working_files/plots/",
    "snwg_forecast_working_files/precomputed/all_dts/"
]

# Check S3 connectivity for all required prefixes
funcs.check_s3_connectivity(s3_bucket, s3_prefixes)
        

#pulling location database
url = "https://raw.githubusercontent.com/noussairlazrak/MLpred/refs/heads/main/global.json"
print(url)
data = json.loads(requests.get(url, stream=True).text)
all_locations = []
force_update = True 
s3_client = boto3.client("s3")
s3_bucket = "s3://localized-forecasts"
#generting the forecasts routine
for key, location_data in list(data.items()):
    #if str(key).lstrip("-").isdigit() and int(key) < -113 and "observation_source" in location_data:
    if location_data.get("observation_source") in ("DoS_Missions", "NASA Pandora", "REMMAQ"):
        site = location_data['location_name'].replace(" ", "_")
        locname = location_data["location_name"]
        lat = location_data["lat"]
        lon = location_data["lon"]
        print(f"Processing Location: {locname} (lat: {lat}, lon: {lon})")
        if location_data["observation_source"] == "DoS_Missions":
            site_file_path = f'./precomputed/all_dts/{site}.json'
            
            # Check if forecast is recent
            if funcs.is_forecast_recent(site_file_path, hours_threshold=5):
                print(f"Skipping {locname} - forecast generated within last 5 hours")
                continue
                
            #generting the forecasts routine for GEOS FP CNN
            site_settings = {'l_name': locname, 
             'species': 'no2', 
             'lat': lat, 
             'lon': lon,
             'silent': True,
             'model_src': 's3',
            }

            try:
                merra2cnn = mlpred.read_merra2_cnn(site=site, frequency = 10, lat = lat, lon = lon)
                metadata = None
                funcs.save_forecast_to_json(merra2cnn, metadata, site_settings=site_settings, species="pm25", sources=["merra2", "geoscf"], output_path=site_file_path)

                # Upload to S3
                funcs.upload_to_s3(site_file_path, s3_client, s3_bucket)
            except Exception as e:
                print(f"Error processing merra 2 forecasts in location {key}: {e}")
                traceback.print_exc()
            
        if location_data["observation_source"] in ("NASA Pandora", "REMMAQ"):
            #generting the forecasts routine for GEOS CF with local observations
            locname = location_data["location_name"]
            file_path = f'./precomputed/all_dts/{locname}.json'
            
            # Check if forecast is recent
            if funcs.is_forecast_recent(file_path, hours_threshold=5):
                print(f" Skipping {locname} - forecast generated within last 5 hours")
                continue
            
            # Set source-specific parameters
            source_type = location_data["observation_source"]
            if source_type == "NASA Pandora":
                obs_src = 'pandora'
                col_name = 'pandora'
                time_col = 'time'
                date_format = '%Y-%m-%d %H:%M'
                obs_val_col = 'Raw Conc.'
            else:  # REMMAQ
                obs_src = 'local'
                col_name = 'local'
                time_col = location_data["obs_options"]["no2"]["time_col"]
                date_format = location_data["obs_options"]["no2"]["time_parser"]
                obs_val_col = location_data["obs_options"]["no2"]["val_col"]
            
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
                 'obs_url': f'REMMAQ/{location_data["obs_options"]["no2"]["file"]}',
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

                forecasts_raw, metadata = mlpred.get_localised_forecast(site_settings=site_settings, obs_settings=obs_settings)
                col_map = {"time": "time", "no2": "no2", "localised": "corrected", "value": col_name}
                fcast = forecasts_raw[["time", "no2", "localised", "value", "o3", "pm25_rh35_gcc", "rh","t10m","tprec","hcho"]].rename(columns=col_map)
                fcast.iloc[:, 1:] = fcast.iloc[:, 1:].clip(lower=0)

                start, end = fcast["time"].min(), fcast["time"].max()
                hourly_index = pd.date_range(start, end, freq="1H")
                aq_df = pd.DataFrame({"time": hourly_index})

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

                fcast = fcast.set_index("time").reindex(hourly_index).reset_index().rename(columns={"index": "time"})
                merg = mlpred.merge_dataframes([aq_df, fcast], "time", resample="1h", how="outer")

                for col in ["no2", "corrected", col_name, "avg"]:
                    merg[col] = merg.get(col, np.nan)

                merg["corrected"] = merg["corrected"].apply(lambda x: x if x > 0 else 0.1)
                merg["avg"] = merg["avg"]*1000
                merg[merg.select_dtypes(include=["float", "int"]).columns] = merg.select_dtypes(include=["float", "int"]).round(2)

                merg_plot = merg.set_index("time").resample("5D").mean()
                merg_plot = merg_plot.reset_index()
                mlpred.gen_plot(merg_plot, [['no2','corrected', col_name,'avg']], [['black','grey', 'red','green']], [['--','-','-','-']], 'NO2', [f'{locname}'], sv_pth=f'./plots/{locname}.png' , resample='1D')

                cutoff = fcast["time"].max() - pd.DateOffset(months=12)
                merg = merg[merg["time"] >= cutoff]
                merg = mlpred.convert_times_column(merg, 'time', lat, lon)
                
                species_map = { 'PM2.5': 'pm25_rh35_gcc', 'NO2': 'corrected', 'O3': 'o3' } 
                avg_hours = { 'NO2': 3, 'O3': 1 }

                merg = funcs.calculate_nowcast(merg, species_columns=species_map, avg_hours=avg_hours)
                print(merg.columns)

                funcs.save_forecast_to_json(merg, metadata, site_settings=site_settings, species="no2",sources=["geoscf", col_name], output_path=file_path)

                # Upload to S3
                funcs.upload_to_s3(file_path, s3_client, s3_bucket)

            except Exception as e:
                print(f"Error processing {source_type} location {key}: {e}")
                traceback.print_exc()
