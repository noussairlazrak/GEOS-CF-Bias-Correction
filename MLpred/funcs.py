import plotly.express as px
import plotly.graph_objects as go
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_error, median_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import json
import sys
import os
import numpy as np
import datetime as dt
import pandas as pd
import requests
import re
from datetime import datetime, timedelta
import datetime as dt
from MLpred import mlpred
import math


BREAKPOINTS = {
    'PM2.5': [
        {"c_low": 0.0, "c_high": 12.0, "i_low": 0, "i_high": 50},
        {"c_low": 12.1, "c_high": 35.4, "i_low": 51, "i_high": 100},
        {"c_low": 35.5, "c_high": 55.4, "i_low": 101, "i_high": 150},
        {"c_low": 55.5, "c_high": 150.4, "i_low": 151, "i_high": 200},
        {"c_low": 150.5, "c_high": 250.4, "i_low": 201, "i_high": 300},
        {"c_low": 250.5, "c_high": 350.4, "i_low": 301, "i_high": 400},
        {"c_low": 350.5, "c_high": 500.4, "i_low": 401, "i_high": 500},
    ],
    'O3': [
            {"c_low": 0, "c_high": 54, "i_low": 0, "i_high": 50},
            {"c_low": 55, "c_high": 70, "i_low": 51, "i_high": 100},
            {"c_low": 71, "c_high": 85, "i_low": 101, "i_high": 150},
            {"c_low": 86, "c_high": 105, "i_low": 151, "i_high": 200},
            {"c_low": 106, "c_high": 200, "i_low": 201, "i_high": 300}
        ],
    'NO2': [
        {"c_low": 0, "c_high": 53, "i_low": 0, "i_high": 50},
        {"c_low": 54, "c_high": 100, "i_low": 51, "i_high": 100},
        {"c_low": 101, "c_high": 360, "i_low": 101, "i_high": 150},
        {"c_low": 361, "c_high": 649, "i_low": 151, "i_high": 200},
        {"c_low": 650, "c_high": 1249, "i_low": 201, "i_high": 300},
        {"c_low": 1250, "c_high": 1649, "i_low": 301, "i_high": 400},
        {"c_low": 1650, "c_high": 2049, "i_low": 401, "i_high": 500},
    ]
}



def calculate_error(predictions):
    """
    Calculate the absolute error associated with prediction intervals

    :param predictions: dataframe of predictions
    :return: None, modifies the prediction dataframe

    """
    predictions['absolute_error_lower'] = (predictions['lower'] - predictions["value"]).abs()
    predictions['absolute_error_upper'] = (predictions['upper'] - predictions["value"]).abs()

    predictions['absolute_error_interval'] = (predictions['absolute_error_lower'] + predictions['absolute_error_upper']) / 2
    predictions['absolute_error_mid'] = (predictions['mid'] - predictions["value"]).abs()

    predictions['in_bounds'] = predictions["value"].between(left=predictions['lower'], right=predictions['upper'])

    return predictions

def show_metrics(metrics):
    """
    Make a boxplot of the metrics associated with prediction intervals

    :param metrics: dataframe of metrics produced from calculate error 
    :return fig: plotly figure
    """
    percent_in_bounds = metrics['in_bounds'].mean() * 100
    metrics_to_plot = metrics[[c for c in metrics if 'absolute_error' in c]]

  
    metrics_to_plot.columns = [column.split('_')[-1].title() for column in metrics_to_plot]


    fig = px.box(
        metrics_to_plot.melt(var_name="metric", value_name='Absolute Error'),
        x="metric",
        y="Absolute Error",
        color='metric',
        title=f"Error Metrics Boxplots    In Bounds = {percent_in_bounds:.2f}%",
        height=800,
        width=1000,
        points=False,
    )


    d = []

    for trace in fig.data:
  
        trace['showlegend'] = False
        d.append(trace)


    fig.data = d
    fig['layout']['font'] = dict(size=20)
    return fig

def confidence_interval_error(CI_DF):
    CI_DF["upper_error"] =  CI_DF.upper - CI_DF.value 
    CI_DF["lower_error"] =  CI_DF.value - CI_DF.lower
    lower_count = (CI_DF["upper_error"] < 0).sum().sum()
    upper_count = (CI_DF["lower_error"] < 0).sum().sum()

    count = lower_count + upper_count
    total_collum = CI_DF.shape[0]

    print ("Error percentage: {:0.2f} %".format(count/total_collum * 100))
    
def make_intervals_box_plot(CI_DF):
    predictions = calculate_error(data)
    metrics = predictions[['absolute_error_lower', 'absolute_error_upper', 'absolute_error_interval', 'absolute_error_mid', 'in_bounds']].copy()
    metrics.describe()
    error_plots = show_metrics(metrics)
    return error_plots



def local_openaq_sites():
    sites_codes=["10812","739"]
    return json.dumps(sites_codes)



def read_openaq_test(url):
        reference_grade_only=True
        silent=False
        remove_outlier=0
        '''Helper routine to read OpenAQ via API (from given url) and create a dataframe of the data'''
        r = requests.get( url )
        if (r.status_code !=200):
            print('Error:  {}'.format(r))
            return None
        allobs = pd.json_normalize(r.json()['results'])
        if allobs.shape[0]==0:
            if not silent:
                print('Warning: no OpenAQ data found for specified url')
            return None

        try:
            allobs = allobs.loc[(allobs['value']>=0.0)&(~np.isnan(allobs['value']))].copy()
            if reference_grade_only:
                allobs = allobs.loc[allobs['sensorType']=='reference grade'].copy()
            allobs['time'] = [dt.datetime.strptime(i,'%Y-%m-%dT%H:%M:%S+00:00') for i in allobs['date.utc']]
            if remove_outlier > 0:
                std = allobs['value'].std()
                mn  = allobs['value'].mean()
                minobs = mn - remove_outlier*std
                maxobs = mn + remove_outlier*std
                norig = allobs.shape[0]
                allobs = allobs.loc[(allobs['value']>=minobs)&(allobs['value']<=maxobs)].copy()
                if not silent:
                    nremoved = norig - allobs.shape[0]
                    print('removed {:.0f} of {:.0f} values because considered outliers ({:.2f}%)'.format(nremoved,norig,np.float(nremoved)/np.float(norig)*100.0))
            return allobs.to_json(orient='records')[1:-1].replace('},{', '} {')

        except:
            if not silent:
                print('Warning ...')
            return None
        
        



eligible_locations = {'Cairo', 'Helsinki', 'Toulouse', 'Hamburg', 'Lagos', 'San_Jose', 'Ceiba__Vega_Baja', 'Nuuk', 'Guadalajara', 'Nuevo_Laredo', 'Ashgabat', 'Algiers', 'Montreal', 'Quito', 'Freetown', 'Dubai', 'Dakar', 'Madrid', 'Kuala_Lumpur', 'Amsterdam', 'Havana', 'Ouagadougou', 'Santo_Domingo', 'Sydney', 'Ankara', 'Busan', 'Calgary', 'Vientiane', 'Ulaanbaatar', 'Port_au_Prince', 'Tel_Aviv', 'Quebec', 'Podgorica', 'CATANO', 'Toronto', 'Valletta', 'Tripoli', 'Nay_Pyi_Taw', 'Chennai', 'Sapporo', 'Wellington', 'Melbourne', 'Gaziantep', 'Kyiv', 'Merida', 'Istanbul', 'Nouakchott', 'Pristina', 'Hanoi', 'Tallinn', 'Warsaw', 'Douala', 'Dhaka', 'Panama_City', 'Durban', 'Hong_Kong', 'Sofia', 'Apia', 'Fukuoka', 'Reykjavik', 'Recife', 'Naha', 'Singapore_City', 'Lisbon', 'Porto_Alegre', 'Stockholm', 'Lusaka', 'Ponta_Delgada', 'Monrovia', 'Florence', 'Canberra', 'Kampala', 'Gaborone', 'Jakarta', 'Nassau', 'Brazzaville', 'Bamako', 'Dili', 'Munchen', 'Asuncion', 'Baghdad', 'Baku', 'Ciudad_Juarez', 'Lyon', 'Erbil', 'Malabo', 'Oslo', 'Phnom_Penh', 'Rio_de_Janeiro', 'Casablanca', 'Brussels', 'Kigali', 'Washington', 'Halifax', 'Chiang_Mai', 'Athens', 'Prague', 'Addis_Ababa', 'Belmopan', 'Tokyo', 'Dushanbe', 'Geneva', 'Kuwait_City', 'Bordeaux', 'Kolkata', 'Amman', 'Kirkuk', 'Pretoria', 'Bangui', 'Banja_Luka', 'Hamilton', 'Osaka', 'Bratislava', 'New_York', 'Dhahran', 'Kingston', 'Belfast', 'Harare', 'Shanghai', 'Tirana', 'Karachi', 'Winnipeg', 'Johannesburg', 'Abuja', 'Tijuana', 'Berlin', 'Kinshasa', 'Zagreb', 'Doha', 'La_Paz', 'Bandar_Seri_Begawan', 'Peshawar', 'Beirut', 'Perth', 'Juba', 'Managua', 'Port_Louis', 'Paramaribo', 'Rangoon', 'Vancouver', 'Rabat', 'Guangzhou', 'Bucharest', 'Manila', 'Mogadishu', 'Colombo', 'Vatican_City', 'Dublin', 'Hyderabad', 'Abu_Dhabi', 'Tegucigalpa', 'Almaty', 'Bogota', 'Vilnius', 'Antananarivo', 'Asmara', 'Leipzig', 'Guayaquil', 'Shenyang', 'Sarajevo', 'Vienna', 'Antwerp', 'Ljubljana', 'Bujumbura', 'Dar_es_Salaam', 'Kathmandu', 'London', 'Buenos_Aires', 'Paris', 'Tunis', 'Djibouti_City', 'Frankfurt_am_Main', 'Cape_Town', 'Edinburgh', 'New_Delhi', 'Luxembourg_City', 'Saint_George_s', 'Cotonou', 'Sao_Paulo', 'Monterrey', 'Mumbai', 'Copenhagen', 'Praia', 'Naples', 'Georgetown', 'Saint_Petersburg', 'Bern', 'Beijing', 'Maputo', 'Santiago', 'Skopje', 'Thessaloniki', 'Maseru', 'Nairobi', 'Dusseldorf', 'Minsk', 'Barcelona', 'Marseille', 'Mostar', 'Jeddah', 'Luanda', 'Rennes', 'Krakow', 'Yaounde', 'Matamoros', 'Seoul', 'Caracas', 'Lahore', 'Kaohsiung', 'Niamey', 'Nicosia', 'Ottawa', 'Curacao', 'Manama', 'Port_of_Spain', 'Auckland', 'Port_Moresby', 'Libreville', 'Vladivostok', 'Nogales', 'Mexico_City', 'Suva', 'Nagoya', 'Belgrade', 'The_Hague', 'Surabaya', 'Taipei', 'Chisinau', 'Wuhan', 'Rome', 'Tbilisi', 'Riga', 'Windhoek', 'Accra', 'Budapest', 'Lilongwe', 'Lima', 'Muscat', 'Riyadh', 'Milan', 'San_Salvador', 'Nur_Sultan', 'Bishkek', 'Jerusalem', 'Yokohama', 'Yerevan', 'Medan', 'Mbabane', 'Banjul', 'Cartagena', 'Hermosillo', 'Guatemala_City', 'Yekaterinburg', 'Tashkent', 'Strasbourg', 'CULEBRA', 'Khartoum', 'Conakry', 'Moscow', 'N_Djamena', 'Bridgetown', 'Alexandria', 'Lome', 'Ho_Chi_Minh_City', 'Bangkok', 'Brasilia', 'Montevideo', 'Islamabad', 'Abidjan', 'Adana' }

def normalize_location_name(name):
    """Normalize location names by replacing spaces and dashes with underscores."""
    return name.replace(' ', '_').replace('-', '_')

def read_local_file(file_path):
    """Reads content from a local text file."""
    with open(file_path, 'r') as file:
        return file.read()

def extract_filenames(text, pattern):
    """Extract filenames matching the given pattern."""
    return re.findall(pattern, text)

def fetch_json_data(url):
    """Fetch JSON data from the given URL."""
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

import json

def get_airnow_sites(file_path="airnow_lists.txt", save_to_file=False, output_file="output.json"):
    """
    Fetches AirNow site data, processes it, and returns JSON. Optionally saves it to a file.

    Args:
        file_path (str): Path to the input file containing filenames.
        save_to_file (bool): Whether to save the result to a file.
        output_file (str): Path to the output file if saving is enabled.

    Returns:
        dict: A dictionary containing processed AirNow site data.
    """
    text = read_local_file(file_path)

    filename_pattern = r'\b[a-zA-Z0-9]*[a-zA-Z]+[a-zA-Z0-9]*\d+[a-zA-Z0-9]*\.json\b'

    filenames = extract_filenames(text, filename_pattern)

    base_url = "https://s3-us-west-1.amazonaws.com//files.airnowtech.org/airnow/recenttrends/Sites/"

    result = {}

    for filename in filenames:
        url = base_url + filename
        try:
            json_data = fetch_json_data(url)

            site_name = json_data.get("siteName", "")
            normalized_site_name = normalize_location_name(site_name)

            if normalized_site_name in eligible_locations:
                station_id = json_data.get("stationID")
                coordinates = json_data.get("coordinates", [None, None])
                lat, lon = coordinates if len(coordinates) == 2 else (None, None)
                monitors = json_data.get("monitors", [])
                monitor_data = {}
                for monitor in monitors:
                    parameter_name = monitor.get("parameterName", "").lower()

                    if "pm2.5" in parameter_name:
                        parameter_name = "pm25"

                    monitor_data[parameter_name] = {
                        "forecasts": f"{site_name}.json",
                        "historical": f"{site_name}.json"
                    }

                result[station_id] = {
                    "location_name": site_name,
                    "lat": lat,
                    "lon": lon,
                    "status": "na",
                    "observation_source": "AirNow",
                    "precomputed_forecasts": monitor_data
                }

        except Exception as e:
            print(f"Error fetching data for {filename}: {e}")


    result_json = json.dumps(result, indent=4)


    if save_to_file:
        try:
            with open(output_file, 'w') as f:
                f.write(result_json)
            print(f"Data saved to {output_file}")
        except Exception as e:
            print(f"Error saving data to file: {e}")


    return result



def merge_jsons(json1_source, json2_source, save_path=None):
    def load_json(source):
        if isinstance(source, str) and source.startswith("http"):
            try:
                response = requests.get(source)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                raise ValueError(f"Failed to fetch JSON from URL: {e}")
        
        elif isinstance(source, str):
            try:
                with open(source, 'r') as file:
                    return json.load(file)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                raise ValueError(f"Failed to load JSON from file: {e}")
        
        elif isinstance(source, dict):
            return source
        
        else:
            raise ValueError("Invalid JSON source. Must be a URL, file path, or dictionary.")

    def merge_dicts(dict1, dict2):
        for key in dict2:
            if key in dict1 and isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                merge_dicts(dict1[key], dict2[key])
            else:
                dict1[key] = dict2[key]
        return dict1


    json1 = load_json(json1_source)
    json2 = load_json(json2_source)

    merged_json = merge_dicts(json1, json2)


    if save_path:
        try:
            with open(save_path, 'w') as file:
                json.dump(merged_json, file, indent=4)
        except IOError as e:
            raise ValueError(f"Failed to save merged JSON to file: {e}")

    return merged_json


def plot_learning_curve(estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(10, 6))
    plt.title("Learning Curves")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='r2')
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    return plt

def calculate_metrics(y_train, y_train_pred, y_test, y_test_pred, json_output=False):
    training_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    training_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    training_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print(f" Bias Correction Metrics:")
    print(f"Training R2: {training_r2:.4f}")
    print(f"Test R2: {test_r2:.4f}")
    print(f"Training MSE: {training_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Training MAE: {training_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")

    metrics_array = np.array([
        ["Training R2", training_r2],
        ["Test R2", test_r2],
        ["Training MSE", training_mse],
        ["Test MSE", test_mse],
        ["Training MAE", training_mae],
        ["Test MAE", test_mae]
    ])


    if json_output:
        metrics_json = {
            "metrics": [
                {"name": metrics_array[0], "value": metrics_array[1]} for metrics_array in metrics_array
            ]
        }
        return json.dumps(metrics_json, indent=4)
    else:
        return metrics_array
    
    

def save_forecast_to_json(merged_data = None, metrics = None, site_settings = None, species = None, sources = None, output_path="forecast.json"):
    """
    Save forecast data and metrics to a JSON file, safely handling missing fields.

    Parameters
    ----------
    merged_data : pandas.DataFrame
        DataFrame containing forecast data, expected to have columns like 'time', 'local_time', pollutants, etc.
    metrics : list or dict
        Performance metrics, expected to contain r2, rmse, mae in order or as keys.
    site_settings : dict
        Dictionary containing site information like 'l_name', 'lat', 'lon'.
    species : str
        Species name.
    output_path : str
        Path to save the JSON output.
    """

    forecasts = merged_data.reset_index().to_dict(orient="records") if not merged_data.empty else []

    latest_training_time = datetime.today().strftime("%Y-%m-%d %H:00:00")


    if not merged_data.empty and "time" in merged_data.columns:
        start_date = merged_data["time"].min().strftime('%Y-%m-%d %H:%M:%S')
        end_date = merged_data["time"].max().strftime('%Y-%m-%d %H:%M:%S')
    else:
        start_date = "N/A"
        end_date = "N/A"

    def safe_get(row, key, datetime_format='%Y-%m-%d %H:%M:%S'):
        val = row.get(key)
        if isinstance(val, datetime):
            return val.strftime(datetime_format)
        return val if val is not None else "N/A"

    forecast_list = []
    for row in forecasts:
        forecast_entry = {}


        if "time" in row:
            forecast_entry["time"] = safe_get(row, "time")
        if "local_time" in row:
            forecast_entry["local_time"] = safe_get(row, "local_time")

        optional_keys = [
            "no2", "corrected", "NO2_AQI", "pandora",
            "o3", "O3_AQI", "avg", "pm25_rh35_gcc",
            "PM25_NowCast_AQI", "rh", "t10m", "tprec", "hcho", "pm25_conc_cnn","pm25_aqi"]
        key_map = {
            "NO2_AQI": "no2_aqi",
            "O3_AQI": "o3_aqi",
            "avg": "openaq",
            "pm25_rh35_gcc": "pm25",
            "PM25_NowCast_AQI": "pm25_aqi"
        }

        for key in optional_keys:
            if key in row:
                json_key = key_map.get(key, key)
                value = row[key]
                if isinstance(value, (float, int)):
                    if isinstance(value, float) and math.isnan(value):
                        value = None 
                    else:
                        value = round(value, 2)

                forecast_entry[json_key] = value

        forecast_list.append(forecast_entry)


    metric_names = ["r2", "rmse", "mae"]
    performance_metrics = []

    if isinstance(metrics, dict):
        for name in metric_names:
            value = str(metrics.get(name, "N/A")) if metrics else "N/A"
            performance_metrics.append({"name": name, "value": value})
    elif isinstance(metrics, (list, tuple)):
        for i, name in enumerate(metric_names):
            if len(metrics) > i and metrics[i] is not None:
                value = str(metrics[i])
            else:
                value = "N/A"
            performance_metrics.append({"name": name, "value": value})
    else:
        # If metrics is None or unexpected type
        for name in metric_names:
            performance_metrics.append({"name": name, "value": "N/A"})
            
    # Normalize sources parameter to a list of strings
    if sources is None:
        sources_list = []
    elif isinstance(sources, str):
        sources_list = [sources]
    elif isinstance(sources, (list, tuple)):
        sources_list = list(sources)
    else:
        sources_list = []

    json_output = {
        "location": site_settings.get("l_name", "N/A"),
        "lat": site_settings.get("lat", "N/A"),
        "lon": site_settings.get("lon", "N/A"),
        "species": species,
        "timezone": merged_data["timezone"].iloc[0] if "timezone" in merged_data.columns and not merged_data.empty else "N/A",
        "message": "200",
        "status": "200",
        "latest_update": latest_training_time,
        "sources": sources_list,
        "metrics": {
            "validation_score": "N/A",
            "total_observation": len(merged_data) if merged_data is not None else 0,
            "performance": {
                "metrics": performance_metrics
            },
            "start_date": start_date,
            "end_date": end_date
        },
        "forecasts": forecast_list
    }

    with open(output_path, "w") as json_file:
        json.dump(json_output, json_file, indent=4)

    print(f"Forecast saved to {output_path}")

    

def openaq_hourly_avgs(sensor_ids, start_date, end_date, silent=False, data_dir='./OPENAQ'):
    os.makedirs(data_dir, exist_ok=True)
    dfs = []

    for sid in sensor_ids:
        csv_path = os.path.join(data_dir, f'sensor_{sid}.csv')

        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path, parse_dates=['time'])
            last_time = existing_df['time'].max()
            new_start_date = max(pd.to_datetime(start_date), last_time)
        else:
            existing_df = pd.DataFrame(columns=['time', 'value'])
            new_start_date = pd.to_datetime(start_date)

        if new_start_date < pd.to_datetime(end_date):
            new_data = mlpred.read_openaq(sid, start=new_start_date, end=end_date, silent=silent, chunk_days=90)
            new_data = new_data[['time', 'value']]
            new_data['time'] = pd.to_datetime(new_data['time'])

            combined_df = pd.concat([existing_df, new_data], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset='time').sort_values('time')
            combined_df.to_csv(csv_path, index=False)
        else:
            combined_df = existing_df

        # Check if this sensor's data is empty
        print(combined_df)
        if combined_df.empty:
            print(f"Sensor {sid} has no data. Aborting and returning an empty DataFrame.")
            return pd.DataFrame(columns=['time', 'avg', 'unit'])

        dfs.append(combined_df)

    # Final check before concat
    if not dfs:
        print("No dataframes to merge. Returning empty DataFrame.")
        return pd.DataFrame(columns=['time', 'avg', 'unit'])

    all_data = pd.concat(dfs, ignore_index=True)
    all_data['time'] = pd.to_datetime(all_data['time'])

    hourly_avg = (
        all_data.groupby(pd.Grouper(key='time', freq='H'))['value']
        .mean()
        .reset_index()
        .rename(columns={'value': 'avg'})
    )

    hourly_avg['unit'] = 'µg/m³'

    return hourly_avg[['time', 'avg', 'unit']]




def vectorized_aqi(concs, breakpoints = BREAKPOINTS):
    aqi = np.full(concs.shape, np.nan)
    for bp in breakpoints:
        mask = (concs >= bp['c_low']) & (concs <= bp['c_high'])
        aqi[mask] = ((bp['i_high'] - bp['i_low']) / (bp['c_high'] - bp['c_low'])) * (concs[mask] - bp['c_low']) + bp['i_low']
    aqi[concs > breakpoints[-1]['c_high']] = 500
    aqi[concs < breakpoints[0]['c_low']] = 0
    return np.round(aqi)

def calculate_nowcast_pm25(pm25_values):
    n = 12
    nowcast = np.full(pm25_values.shape, np.nan)
    for i in range(len(pm25_values)):
        start_idx = max(0, i - n + 1)
        window = pm25_values[start_idx:i+1]
        if len(window) == 0:
            continue
        c_min = np.min(window)
        c_max = np.max(window)
        if c_max == 0:
            nowcast[i] = 0
            continue
        range_c = c_max - c_min
        scaled_rate = range_c / c_max if c_max != 0 else 0
        weight_factor = max(0.5, 1 - scaled_rate)
        weights = np.array([weight_factor ** (len(window) - 1 - j) for j in range(len(window))])
        nowcast[i] = np.sum(weights * window) / np.sum(weights)
    return nowcast

def calculate_nowcast(df, species_columns=None, avg_hours=None):
    """
    Calculate NowCast AQI for species in df, allowing custom column names.
    
    Parameters:
    - df: pandas DataFrame with pollutant concentration columns.
    - species_columns: dict mapping species names ('PM2.5', 'NO2', 'O3') to df column names.
                       Example: {'PM2.5': 'pm25_conc', 'NO2': 'no2_ppb', 'O3': 'ozone_ppb'}
                       If None, defaults to species names as columns.
    - avg_hours: dict specifying averaging window hours per species (for NO2, O3).
                 PM2.5 always uses NowCast method on raw hourly data.
    
    Returns:
    - df with added NowCast/AQI columns named:
      - PM2.5: 'PM2.5_NowCast_Concentration', 'PM2.5_NowCast_AQI'
      - Others: '{species}_AQI'
    """
    df = df.copy()
    if species_columns is None:
        species_columns = {'PM2.5': 'PM2.5', 'NO2': 'NO2', 'O3': 'O3'}
    if avg_hours is None:
        avg_hours = {}

    for species, col in species_columns.items():
        if col not in df.columns:
            continue 

        if species == 'PM2.5':
            pm25_vals = df[col].values
            nowcast_pm25 = calculate_nowcast_pm25(pm25_vals)
            df['PM25_NowCast_Concentration'] = nowcast_pm25
            df['PM25_NowCast_AQI'] = vectorized_aqi(nowcast_pm25, BREAKPOINTS['PM2.5'])
        else:
            window = avg_hours.get(species, 1)
            rolling_avg = df[col].rolling(window=window, min_periods=1).mean()
            df[f'{species}_AQI'] = vectorized_aqi(rolling_avg.values, BREAKPOINTS[species])

    return df