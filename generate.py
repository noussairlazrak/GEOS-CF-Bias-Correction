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
import datetime as dt

import boto3
from botocore.exceptions import ClientError

# S3 bucket and prefixes to check
s3_bucket = "smce-geos-cf-forecasts-oss-shared"
s3_prefixes = [
    "snwg_forecast_working_files/GEOS_CF/",
    "snwg_forecast_working_files/OPENAQ/",
    "snwg_forecast_working_files/plots/",
    "snwg_forecast_working_files/precomputed/all_dts/"
]

def check_s3_access(bucket, prefix):
    s3 = boto3.client("s3")
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        if "Contents" in response:
            print(f"✅ Access confirmed for s3://{bucket}/{prefix}")
            return True
        else:
            print(f"(X) No files found, but access confirmed for s3://{bucket}/{prefix}")
            return True
    except ClientError as e:
        print(f"(X) Access denied or error for s3://{bucket}/{prefix}: {e}")
        return False

# Check S3 connectivity for all required prefixes
for prefix in s3_prefixes:
    check_s3_access(s3_bucket, prefix)
        

#pulling location database
url = "https://raw.githubusercontent.com/noussairlazrak/MLpred/refs/heads/main/global.json"
print(url)
data = json.loads(requests.get(url, stream=True).text)
all_locations = []
force_update = True 
s3_client = boto3.client("s3")
s3_bucket = "s3://smce-geos-cf-forecasts-oss-shared"
#generting the forecasts routine
for key, location_data in list(data.items()):
    #if str(key).lstrip("-").isdigit() and int(key) < -113 and "observation_source" in location_data:
    if location_data.get("observation_source") in ("DoS_Missions", "NASA Pandora"):
        site = location_data['location_name'].replace(" ", "_")
        locname = location_data["location_name"]
        lat = location_data["lat"]
        lon = location_data["lon"]
        print(f"Processing Location: {locname} (lat: {lat}, lon: {lon})")
        if location_data["observation_source"] == "DoS_Missions":
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
                site_file_path = f'./precomputed/all_dts/{site}.json'
                funcs.save_forecast_to_json(merra2cnn, metadata, site_settings=site_settings, species="pm25", sources=["merra2", "geoscf"], output_path=site_file_path)

                # S3 verifs
                if os.path.exists(site_file_path):
                    print(f"Forecast file {site_file_path} exists. Uploading to S3...")
                    s3_bucket_name = s3_bucket.replace("s3://", "")
                    s3_key = f"snwg_forecast_working_files/precomputed/all_dts/{os.path.basename(site_file_path)}"
                    try:
                        s3_client.upload_file(site_file_path, s3_bucket_name, s3_key)
                        print(f"Successfully uploaded {site_file_path} to s3://{s3_bucket_name}/{s3_key}")
                    except Exception as upload_err:
                        print(f"Failed to upload {site_file_path} to S3: {upload_err}")
                else:
                    print(f"Forecast file {site_file_path} does not exist. Skipping S3 upload.")
            except Exception as e:
                print(f"Error processing merra 2 forecasts in location {key}: {e}")
            
        if location_data["observation_source"] == "NASA Pandora":
            #generting the forecasts routine for GEOS CF PANDORA
            locname = location_data["location_name"]
            file_path = f'./precomputed/all_dts/{locname}.json'
            
            try:
                site_settings = {'l_name': locname, 
                 'species': 'no2', 
                 'lat': lat, 
                 'lon': lon,
                 'silent': True,
                 'model_src': 's3',
                 'obs_src': 'pandora',
                 'openaq_id': None,
                 'model_tuning' : False,
                 'model_url': '#',
                 'obs_url': location_data["obs_options"]["no2"]["file"],
                 'resample' : '1h',
                 'unit' : 'ppb',
                 'interpolation': True,
                 'remove_outlier': True,
                 'start' : dt.datetime(2018, 1, 1),   
                 'end': dt.datetime.today()
                }
                obs_settings = {'time_col': 'time', 
                                     'date_format': '%Y-%m-%d %H:%M', 
                                     'obs_val_col': 'Raw Conc.', 
                                     'lat_col': None, 
                                     'lon_col': None,
                                     'remove_outlier': False,
                                    }

                validation_sets = []

                forecasts_raw, metadata = mlpred.get_localised_forecast(site_settings=site_settings, obs_settings=obs_settings)
                col_map = {"time": "time", "no2": "no2", "localised": "corrected", "value": "pandora"}
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
                


                for col in ["no2", "corrected", "pandora", "avg"]:
                    merg[col] = merg.get(col, np.nan)


                merg["corrected"] = merg["corrected"].apply(lambda x: x if x > 0 else 0.1)
                merg["avg"] = merg["avg"]*1000
                merg[merg.select_dtypes(include=["float", "int"]).columns] = merg.select_dtypes(include=["float", "int"]).round(2)


                merg_plot = merg.set_index("time").resample("5D").mean()
                merg_plot = merg_plot.reset_index()
                mlpred.gen_plot(merg_plot, [['no2','corrected', 'pandora','avg']], [['black','grey', 'red','green']], [['--','-','-','-']], 'NO2', [f'{locname}'], sv_pth=f'./plots/{locname}.png' , resample='1D')



                cutoff = fcast["time"].max() - pd.DateOffset(months=12)
                merg = merg[merg["time"] >= cutoff]
                merg = mlpred.convert_times_column(merg, 'time', lat, lon)
                
                species_map = { 'PM2.5': 'pm25_rh35_gcc', 'NO2': 'corrected', 'O3': 'o3' } 
                avg_hours = { 'NO2': 3, 'O3': 1 }

                merg = funcs.calculate_nowcast(merg, species_columns=species_map, avg_hours=avg_hours)
                
                print(merg.columns)

                funcs.save_forecast_to_json(merg, metadata, site_settings=site_settings, species="no2",sources=[ "geoscf","pandora"], output_path=file_path)

                # Verify file exists before pushing to S3
                if os.path.exists(file_path):
                    print(f"Forecast file {file_path} exists. Uploading to S3...")
                    s3_bucket_name = s3_bucket.replace("s3://", "")
                    s3_key = f"snwg_forecast_working_files/precomputed/all_dts/{os.path.basename(file_path)}"
                    try:
                        s3_client.upload_file(file_path, s3_bucket_name, s3_key)
                        print(f"Successfully uploaded {file_path} to s3://{s3_bucket_name}/{s3_key}")
                    except Exception as upload_err:
                        print(f"Failed to upload {file_path} to S3: {upload_err}")
                else:
                    print(f"Forecast file {file_path} does not exist. Skipping S3 upload.")

            except Exception as e:

                print(f"Error processing location {key}: {e}")
