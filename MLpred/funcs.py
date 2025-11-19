import json
import math
import os
import re
from datetime import datetime, timedelta

import boto3
import numpy as np
import pandas as pd
import plotly.express as px
import requests
from botocore.exceptions import ClientError
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from MLpred import mlpred


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
    """Calculate absolute error metrics for prediction intervals.

    Parameters
    ----------
    predictions : pd.DataFrame
        DataFrame with columns: 'lower', 'mid', 'upper', 'value'

    Returns
    -------
    pd.DataFrame
        DataFrame with added error columns
    """
    predictions['absolute_error_lower'] = (predictions['lower'] - predictions["value"]).abs()
    predictions['absolute_error_upper'] = (predictions['upper'] - predictions["value"]).abs()
    predictions['absolute_error_interval'] = (predictions['absolute_error_lower'] + predictions['absolute_error_upper']) / 2
    predictions['absolute_error_mid'] = (predictions['mid'] - predictions["value"]).abs()
    predictions['in_bounds'] = predictions["value"].between(left=predictions['lower'], right=predictions['upper'])

    return predictions

def show_metrics(metrics):
    """
    Make a boxplot of the metrics associated with prediction intervals.

    Parameters
    ----------
    metrics : pd.DataFrame
        Dataframe of metrics produced from calculate_error
        
    Returns
    -------
    plotly.graph_objects.Figure
        Box plot figure with error metrics
    """
    percent_in_bounds = metrics['in_bounds'].mean() * 100
    metrics_to_plot = metrics[[c for c in metrics if 'absolute_error' in c]]
    metrics_to_plot.columns = [col.split('_')[-1].title() for col in metrics_to_plot]

    fig = px.box(
        metrics_to_plot.melt(var_name="metric", value_name='Absolute Error'),
        x="metric",
        y="Absolute Error",
        color='metric',
        title=f"Error Metrics Boxplots - In Bounds = {percent_in_bounds:.2f}%",
        height=800,
        width=1000,
        points=False,
    )

    for trace in fig.data:
        trace['showlegend'] = False

    fig['layout']['font'] = dict(size=20)
    return fig

def confidence_interval_error(ci_df):
    """Calculate and print error percentage for confidence intervals."""
    ci_df["upper_error"] = ci_df.upper - ci_df.value
    ci_df["lower_error"] = ci_df.value - ci_df.lower
    lower_count = (ci_df["upper_error"] < 0).sum()
    upper_count = (ci_df["lower_error"] < 0).sum()

    count = lower_count + upper_count
    total_rows = ci_df.shape[0]
    error_percent = count / total_rows * 100
    print(f"Error percentage: {error_percent:.2f}%")
    
def make_intervals_box_plot(predictions_df):
    """Create box plot visualization of prediction interval errors.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with prediction data
        
    Returns
    -------
    plotly.graph_objects.Figure
        Box plot of error metrics
    """
    predictions = calculate_error(predictions_df)
    error_cols = ['absolute_error_lower', 'absolute_error_upper', 'absolute_error_interval', 'absolute_error_mid', 'in_bounds']
    metrics = predictions[[col for col in error_cols if col in predictions.columns]].copy()
    return show_metrics(metrics)



def local_openaq_sites():
    """Return JSON string of local OpenAQ site codes."""
    sites_codes = ["10812", "739"]
    return json.dumps(sites_codes)



def read_openaq_test(url, reference_grade_only=True, silent=False, remove_outlier=0):
    """Read OpenAQ data from API and return as JSON string.
    
    Parameters
    ----------
    url : str
        OpenAQ API URL
    reference_grade_only : bool
        Filter for reference grade sensors only
    silent : bool
        Suppress warning messages
    remove_outlier : int
        Number of standard deviations for outlier removal
        
    Returns
    -------
    str or None
        JSON string of observations or None if error
    """
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f'Error: {response}')
            return None
            
        allobs = pd.json_normalize(response.json()['results'])
        if allobs.empty:
            if not silent:
                print('Warning: no OpenAQ data found for specified URL')
            return None

        # Filter valid data
        allobs = allobs.loc[(allobs['value'] >= 0.0) & (~np.isnan(allobs['value']))].copy()
        
        if reference_grade_only:
            allobs = allobs.loc[allobs['sensorType'] == 'reference grade'].copy()
        
        # Parse timestamps
        allobs['time'] = pd.to_datetime(allobs['date.utc'])
        
        # Remove outliers
        if remove_outlier > 0:
            mean_val = allobs['value'].mean()
            std_val = allobs['value'].std()
            lower_bound = mean_val - remove_outlier * std_val
            upper_bound = mean_val + remove_outlier * std_val
            
            orig_count = len(allobs)
            allobs = allobs.loc[(allobs['value'] >= lower_bound) & (allobs['value'] <= upper_bound)].copy()
            
            if not silent:
                removed_count = orig_count - len(allobs)
                removal_pct = removed_count / orig_count * 100
                print(f'Removed {removed_count:.0f} of {orig_count:.0f} outlier values ({removal_pct:.2f}%)')
        
        return allobs.to_json(orient='records')[1:-1].replace('},{', '} {')

    except Exception as e:
        if not silent:
            print(f'Warning: {e}')
        return None
        
        



eligible_locations = {'Cairo', 'Helsinki', 'Toulouse', 'Hamburg', 'Lagos', 'San_Jose', 'Ceiba__Vega_Baja', 'Nuuk', 'Guadalajara', 'Nuevo_Laredo', 'Ashgabat', 'Algiers', 'Montreal', 'Quito', 'Freetown', 'Dubai', 'Dakar', 'Madrid', 'Kuala_Lumpur', 'Amsterdam', 'Havana', 'Ouagadougou', 'Santo_Domingo', 'Sydney', 'Ankara', 'Busan', 'Calgary', 'Vientiane', 'Ulaanbaatar', 'Port_au_Prince', 'Tel_Aviv', 'Quebec', 'Podgorica', 'CATANO', 'Toronto', 'Valletta', 'Tripoli', 'Nay_Pyi_Taw', 'Chennai', 'Sapporo', 'Wellington', 'Melbourne', 'Gaziantep', 'Kyiv', 'Merida', 'Istanbul', 'Nouakchott', 'Pristina', 'Hanoi', 'Tallinn', 'Warsaw', 'Douala', 'Dhaka', 'Panama_City', 'Durban', 'Hong_Kong', 'Sofia', 'Apia', 'Fukuoka', 'Reykjavik', 'Recife', 'Naha', 'Singapore_City', 'Lisbon', 'Porto_Alegre', 'Stockholm', 'Lusaka', 'Ponta_Delgada', 'Monrovia', 'Florence', 'Canberra', 'Kampala', 'Gaborone', 'Jakarta', 'Nassau', 'Brazzaville', 'Bamako', 'Dili', 'Munchen', 'Asuncion', 'Baghdad', 'Baku', 'Ciudad_Juarez', 'Lyon', 'Erbil', 'Malabo', 'Oslo', 'Phnom_Penh', 'Rio_de_Janeiro', 'Casablanca', 'Brussels', 'Kigali', 'Washington', 'Halifax', 'Chiang_Mai', 'Athens', 'Prague', 'Addis_Ababa', 'Belmopan', 'Tokyo', 'Dushanbe', 'Geneva', 'Kuwait_City', 'Bordeaux', 'Kolkata', 'Amman', 'Kirkuk', 'Pretoria', 'Bangui', 'Banja_Luka', 'Hamilton', 'Osaka', 'Bratislava', 'New_York', 'Dhahran', 'Kingston', 'Belfast', 'Harare', 'Shanghai', 'Tirana', 'Karachi', 'Winnipeg', 'Johannesburg', 'Abuja', 'Tijuana', 'Berlin', 'Kinshasa', 'Zagreb', 'Doha', 'La_Paz', 'Bandar_Seri_Begawan', 'Peshawar', 'Beirut', 'Perth', 'Juba', 'Managua', 'Port_Louis', 'Paramaribo', 'Rangoon', 'Vancouver', 'Rabat', 'Guangzhou', 'Bucharest', 'Manila', 'Mogadishu', 'Colombo', 'Vatican_City', 'Dublin', 'Hyderabad', 'Abu_Dhabi', 'Tegucigalpa', 'Almaty', 'Bogota', 'Vilnius', 'Antananarivo', 'Asmara', 'Leipzig', 'Guayaquil', 'Shenyang', 'Sarajevo', 'Vienna', 'Antwerp', 'Ljubljana', 'Bujumbura', 'Dar_es_Salaam', 'Kathmandu', 'London', 'Buenos_Aires', 'Paris', 'Tunis', 'Djibouti_City', 'Frankfurt_am_Main', 'Cape_Town', 'Edinburgh', 'New_Delhi', 'Luxembourg_City', 'Saint_George_s', 'Cotonou', 'Sao_Paulo', 'Monterrey', 'Mumbai', 'Copenhagen', 'Praia', 'Naples', 'Georgetown', 'Saint_Petersburg', 'Bern', 'Beijing', 'Maputo', 'Santiago', 'Skopje', 'Thessaloniki', 'Maseru', 'Nairobi', 'Dusseldorf', 'Minsk', 'Barcelona', 'Marseille', 'Mostar', 'Jeddah', 'Luanda', 'Rennes', 'Krakow', 'Yaounde', 'Matamoros', 'Seoul', 'Caracas', 'Lahore', 'Kaohsiung', 'Niamey', 'Nicosia', 'Ottawa', 'Curacao', 'Manama', 'Port_of_Spain', 'Auckland', 'Port_Moresby', 'Libreville', 'Vladivostok', 'Nogales', 'Mexico_City', 'Suva', 'Nagoya', 'Belgrade', 'The_Hague', 'Surabaya', 'Taipei', 'Chisinau', 'Wuhan', 'Rome', 'Tbilisi', 'Riga', 'Windhoek', 'Accra', 'Budapest', 'Lilongwe', 'Lima', 'Muscat', 'Riyadh', 'Milan', 'San_Salvador', 'Nur_Sultan', 'Bishkek', 'Jerusalem', 'Yokohama', 'Yerevan', 'Medan', 'Mbabane', 'Banjul', 'Cartagena', 'Hermosillo', 'Guatemala_City', 'Yekaterinburg', 'Tashkent', 'Strasbourg', 'CULEBRA', 'Khartoum', 'Conakry', 'Moscow', 'N_Djamena', 'Bridgetown', 'Alexandria', 'Lome', 'Ho_Chi_Minh_City', 'Bangkok', 'Brasilia', 'Montevideo', 'Islamabad', 'Abidjan', 'Adana' }

def normalize_location_name(name):
    """Normalize location names by replacing spaces and dashes with underscores."""
    return name.replace(' ', '_').replace('-', '_')

def read_local_file(file_path):
    """Reads content from a local text file.
    
    Parameters
    ----------
    file_path : str
        Path to the file to read
        
    Returns
    -------
    str
        File contents, or empty string on error
    """
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return ""
    except (IOError, OSError) as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def extract_filenames(text, pattern):
    """Extract filenames matching the given pattern.
    
    Parameters
    ----------
    text : str
        Text to search for patterns
    pattern : str
        Regular expression pattern to match
        
    Returns
    -------
    list
        List of matched filenames
    """
    try:
        return re.findall(pattern, text)
    except re.error as e:
        print(f"Invalid regex pattern: {e}")
        return []

def fetch_json_data(url):
    """Fetch JSON data from the given URL.
    
    Parameters
    ----------
    url : str
        URL to fetch data from
        
    Returns
    -------
    dict
        Parsed JSON data, or empty dict on error
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.Timeout:
        print(f"Timeout fetching {url}")
        return {}
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return {}
    except ValueError as e:
        print(f"Invalid JSON from {url}: {e}")
        return {}

def get_airnow_sites(file_path="airnow_lists.txt", save_to_file=False, output_file="output.json"):
    """Fetch and process AirNow site data.
    
    Parameters
    ----------
    file_path : str
        Path to file containing AirNow JSON filenames
    save_to_file : bool
        Whether to save result to file
    output_file : str
        Output file path
        
    Returns
    -------
    dict
        Processed AirNow site data by station ID
    """
    try:
        text = read_local_file(file_path)
    except (FileNotFoundError, IOError) as e:
        print(f"Error reading file {file_path}: {e}")
        return {}
        
    filename_pattern = r'\b[a-zA-Z0-9]*[a-zA-Z]+[a-zA-Z0-9]*\d+[a-zA-Z0-9]*\.json\b'
    filenames = extract_filenames(text, filename_pattern)
    base_url = "https://s3-us-west-1.amazonaws.com//files.airnowtech.org/airnow/recenttrends/Sites/"
    
    result = {}
    
    for filename in filenames:
        try:
            json_data = fetch_json_data(base_url + filename)
            site_name = json_data.get("siteName", "")
            normalized_name = normalize_location_name(site_name)
            
            if normalized_name not in eligible_locations:
                continue
            
            station_id = json_data.get("stationID")
            coordinates = json_data.get("coordinates", [None, None])
            lat, lon = (coordinates[0], coordinates[1]) if len(coordinates) == 2 else (None, None)
            
            monitor_data = {}
            for monitor in json_data.get("monitors", []):
                param_name = monitor.get("parameterName", "").lower()
                if "pm2.5" in param_name:
                    param_name = "pm25"
                monitor_data[param_name] = {
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
        except (requests.RequestException, ValueError, KeyError, TypeError) as e:
            print(f"Error processing {filename}: {e}")
    
    if save_to_file:
        try:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=4)
            print(f"Data saved to {output_file}")
        except (IOError, OSError) as e:
            print(f"Error saving to {output_file}: {e}")
    
    return result



def merge_jsons(json1_source, json2_source, save_path=None):
    """Merge two JSON sources (URLs, files, or dicts).
    
    Parameters
    ----------
    json1_source : str or dict
        First JSON source (URL, file path, or dict)
    json2_source : str or dict
        Second JSON source (URL, file path, or dict)
    save_path : str, optional
        Path to save merged result
        
    Returns
    -------
    dict
        Merged JSON dictionary
    """
    def load_json(source):
        """Load JSON from URL, file, or dict."""
        if isinstance(source, dict):
            return source
        if isinstance(source, str) and source.startswith("http"):
            try:
                return requests.get(source).json()
            except (requests.RequestException, ValueError) as e:
                raise ValueError(f"Failed to fetch JSON from URL: {e}")
        if isinstance(source, str):
            try:
                with open(source, 'r') as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
                raise ValueError(f"Failed to load JSON from file: {e}")
        raise ValueError("JSON source must be a URL, file path, or dictionary.")

    def merge_dicts(dict1, dict2):
        """Recursively merge dict2 into dict1."""
        for key, value in dict2.items():
            if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
                merge_dicts(dict1[key], value)
            else:
                dict1[key] = value
        return dict1

    json1 = load_json(json1_source)
    json2 = load_json(json2_source)
    merged_json = merge_dicts(json1, json2)

    if save_path:
        try:
            with open(save_path, 'w') as f:
                json.dump(merged_json, f, indent=4)
        except IOError as e:
            raise ValueError(f"Failed to save merged JSON to file: {e}")

    return merged_json


def plot_learning_curve(estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)):
    """Plot learning curves for model evaluation.
    
    Note: Requires matplotlib and sklearn.model_selection.learning_curve
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.model_selection import learning_curve
    except ImportError as e:
        raise ImportError(f"Required library not found: {e}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    train_sizes_out, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='r2'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    ax.fill_between(train_sizes_out, train_mean - train_std, train_mean + train_std,
                    alpha=0.1, color="r")
    ax.fill_between(train_sizes_out, test_mean - test_std, test_mean + test_std,
                    alpha=0.1, color="g")
    ax.plot(train_sizes_out, train_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes_out, test_mean, 'o-', color="g", label="CV score")
    
    ax.set_title("Learning Curves")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score (R²)")
    ax.grid()
    ax.legend(loc="best")
    
    return fig

def calculate_metrics(y_train, y_train_pred, y_test, y_test_pred, json_output=False):
    """Calculate and return model performance metrics.
    
    Parameters
    ----------
    y_train, y_train_pred : array-like
        Training labels and predictions
    y_test, y_test_pred : array-like
        Test labels and predictions
    json_output : bool
        Return metrics as JSON string if True, else numpy array
        
    Returns
    -------
    str or np.ndarray
        Performance metrics in requested format, or empty result on error
    """
    try:
        metrics = {
            "Training R2": r2_score(y_train, y_train_pred),
            "Test R2": r2_score(y_test, y_test_pred),
            "Training MSE": mean_squared_error(y_train, y_train_pred),
            "Test MSE": mean_squared_error(y_test, y_test_pred),
            "Training MAE": mean_absolute_error(y_train, y_train_pred),
            "Test MAE": mean_absolute_error(y_test, y_test_pred),
        }

        print("Bias Correction Metrics:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        if json_output:
            metrics_json = {
                "metrics": [{"name": k, "value": v} for k, v in metrics.items()]
            }
            return json.dumps(metrics_json, indent=4)
        else:
            return np.array([[k, v] for k, v in metrics.items()])
    except (ValueError, TypeError) as e:
        print(f"Error calculating metrics: {e}")
        if json_output:
            return json.dumps({"metrics": []}, indent=4)
        else:
            return np.array([])
    except Exception as e:
        print(f"Unexpected error in calculate_metrics: {e}")
        if json_output:
            return json.dumps({"metrics": []}, indent=4)
        else:
            return np.array([])
    
    

def _normalize_sources(sources):
    """Convert sources to list of strings."""
    if sources is None:
        return []
    if isinstance(sources, str):
        return [sources]
    if isinstance(sources, (list, tuple)):
        return list(sources)
    return []


def _format_metrics(metrics):
    """Format metrics dict/list/tuple into standardized list."""
    metric_names = ["r2", "rmse", "mae"]
    result = []
    
    if isinstance(metrics, dict):
        for name in metric_names:
            value = str(metrics.get(name, "N/A")) if metrics else "N/A"
            result.append({"name": name, "value": value})
    elif isinstance(metrics, (list, tuple)):
        for i, name in enumerate(metric_names):
            value = str(metrics[i]) if i < len(metrics) and metrics[i] is not None else "N/A"
            result.append({"name": name, "value": value})
    else:
        for name in metric_names:
            result.append({"name": name, "value": "N/A"})
    
    return result


def _process_forecast_row(row, key_map):
    """Extract and format optional forecast fields from row."""
    optional_keys = [
        "no2", "corrected", "NO2_AQI", "pandora",
        "o3", "O3_AQI", "avg", "pm25_rh35_gcc",
        "PM25_NowCast_AQI", "rh", "t10m", "tprec", "hcho", "pm25_conc_cnn", "pm25_aqi"
    ]
    
    entry = {}
    for key in optional_keys:
        if key in row:
            json_key = key_map.get(key, key)
            value = row[key]
            if isinstance(value, float):
                if math.isnan(value):
                    value = None
                else:
                    value = round(value, 2)
            entry[json_key] = value
    
    return entry


def save_forecast_to_json(merged_data=None, metrics=None, site_settings=None, species=None, 
                          sources=None, output_path="forecast.json"):
    """Save forecast data and metrics to JSON file.

    Parameters
    ----------
    merged_data : pd.DataFrame, optional
        Forecast data with 'time', 'local_time', and pollutant columns
    metrics : list, tuple, or dict, optional
        Performance metrics (r2, rmse, mae)
    site_settings : dict, optional
        Site info: 'l_name', 'lat', 'lon'
    species : str, optional
        Pollutant species name
    sources : str, list, or tuple, optional
        Data source(s)
    output_path : str
        Output file path
    """
    if merged_data is None:
        merged_data = pd.DataFrame()
    site_settings = site_settings or {}
    
    forecasts = merged_data.reset_index().to_dict(orient="records") if not merged_data.empty else []
    latest_training_time = datetime.today().strftime("%Y-%m-%d %H:00:00")

    if not merged_data.empty and "time" in merged_data.columns:
        start_date = merged_data["time"].min().strftime('%Y-%m-%d %H:%M:%S')
        end_date = merged_data["time"].max().strftime('%Y-%m-%d %H:%M:%S')
    else:
        start_date = "N/A"
        end_date = "N/A"

    key_map = {
        "NO2_AQI": "no2_aqi",
        "O3_AQI": "o3_aqi",
        "avg": "openaq",
        "pm25_rh35_gcc": "pm25",
        "PM25_NowCast_AQI": "pm25_aqi"
    }

    def safe_format_time(val):
        """Format datetime to ISO format string."""
        if isinstance(val, datetime):
            return val.strftime('%Y-%m-%d %H:%M:%S')
        return val if val is not None else "N/A"

    forecast_list = []
    for row in forecasts:
        entry = _process_forecast_row(row, key_map)
        if "time" in row:
            entry["time"] = safe_format_time(row["time"])
        if "local_time" in row:
            entry["local_time"] = safe_format_time(row["local_time"])
        forecast_list.append(entry)

    try:
        timezone = merged_data["timezone"].iloc[0] if "timezone" in merged_data.columns and not merged_data.empty else "N/A"
    except (IndexError, KeyError, TypeError):
        timezone = "N/A"
    
    json_output = {
        "location": site_settings.get("l_name", "N/A"),
        "lat": site_settings.get("lat", "N/A"),
        "lon": site_settings.get("lon", "N/A"),
        "species": species,
        "timezone": timezone,
        "message": "200",
        "status": "200",
        "latest_update": latest_training_time,
        "sources": _normalize_sources(sources),
        "metrics": {
            "validation_score": "N/A",
            "total_observation": len(merged_data),
            "performance": {"metrics": _format_metrics(metrics)},
            "start_date": start_date,
            "end_date": end_date
        },
        "forecasts": forecast_list
    }

    try:
        with open(output_path, "w") as f:
            json.dump(json_output, f, indent=4)
        print(f"Forecast saved to {output_path}")
    except (IOError, OSError) as e:
        print(f"Error saving forecast to {output_path}: {e}")

    

def openaq_hourly_avgs(sensor_ids, start_date, end_date, silent=False, data_dir='./OPENAQ'):
    """Fetch and aggregate hourly OpenAQ data for sensors.
    
    Parameters
    ----------
    sensor_ids : list
        List of sensor IDs
    start_date : datetime
        Start date for data
    end_date : datetime
        End date for data
    silent : bool
        Suppress warnings
    data_dir : str
        Directory to cache CSV files
        
    Returns
    -------
    pd.DataFrame
        Hourly averaged data with time, avg, unit columns
    """
    os.makedirs(data_dir, exist_ok=True)
    dfs = []

    for sid in sensor_ids:
        try:
            csv_path = os.path.join(data_dir, f'sensor_{sid}.csv')

            if os.path.exists(csv_path):
                try:
                    existing_df = pd.read_csv(csv_path, parse_dates=['time'])
                    last_time = existing_df['time'].max()
                    new_start_date = max(pd.to_datetime(start_date), last_time)
                except (pd.errors.EmptyDataError, KeyError, ValueError) as e:
                    if not silent:
                        print(f"Error reading CSV for sensor {sid}: {e}")
                    existing_df = pd.DataFrame(columns=['time', 'value'])
                    new_start_date = pd.to_datetime(start_date)
            else:
                existing_df = pd.DataFrame(columns=['time', 'value'])
                new_start_date = pd.to_datetime(start_date)

            if new_start_date < pd.to_datetime(end_date):
                new_data = mlpred.read_openaq(sid, start=new_start_date, end=end_date, silent=silent, chunk_days=90)
                
                if new_data is None or new_data.empty:
                    combined_df = existing_df
                else:
                    try:
                        new_data = new_data[['time', 'value']]
                        new_data['time'] = pd.to_datetime(new_data['time'])
                        combined_df = pd.concat([existing_df, new_data], ignore_index=True)
                        combined_df = combined_df.drop_duplicates(subset='time').sort_values('time')
                        combined_df.to_csv(csv_path, index=False)
                    except (KeyError, TypeError) as e:
                        if not silent:
                            print(f"Error processing OpenAQ data for sensor {sid}: {e}")
                        combined_df = existing_df
            else:
                combined_df = existing_df

            # Check if this sensor's data is empty
            if combined_df.empty:
                if not silent:
                    print(f"Sensor {sid} has no data.")
                continue

            dfs.append(combined_df)
            
        except Exception as e:
            if not silent:
                print(f"Error processing sensor {sid}: {e}")
            continue

    # Final check before concat
    if not dfs:
        if not silent:
            print("No dataframes to merge. Returning empty DataFrame.")
        return pd.DataFrame(columns=['time', 'avg', 'unit'])

    try:
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
        
    except Exception as e:
        if not silent:
            print(f"Error creating hourly averages: {e}")
        return pd.DataFrame(columns=['time', 'avg', 'unit'])




def vectorized_aqi(concs, breakpoints = BREAKPOINTS):
    aqi = np.full(concs.shape, np.nan)
    for bp in breakpoints:
        mask = (concs >= bp['c_low']) & (concs <= bp['c_high'])
        aqi[mask] = ((bp['i_high'] - bp['i_low']) / (bp['c_high'] - bp['c_low'])) * (concs[mask] - bp['c_low']) + bp['i_low']
    aqi[concs > breakpoints[-1]['c_high']] = 500
    aqi[concs < breakpoints[0]['c_low']] = 0
    return np.round(aqi)

def calculate_nowcast_pm25(pm25_values, window_size=12):
    """Calculate NowCast PM2.5 concentration.
    
    Parameters
    ----------
    pm25_values : array-like
        PM2.5 concentration values
    window_size : int
        Lookback window size (default 12 hours)
        
    Returns
    -------
    np.ndarray
        NowCast values
    """
    nowcast = np.full(pm25_values.shape, np.nan)
    
    for i in range(len(pm25_values)):
        start_idx = max(0, i - window_size + 1)
        window = pm25_values[start_idx:i + 1]
        
        if len(window) == 0:
            continue
        
        c_min = np.min(window)
        c_max = np.max(window)
        
        if c_max == 0:
            nowcast[i] = 0
            continue
        
        # Calculate weight factor based on range
        range_c = c_max - c_min
        scaled_rate = range_c / c_max
        weight_factor = max(0.5, 1 - scaled_rate)
        
        # Apply exponential weights
        weights = np.array([weight_factor ** (len(window) - 1 - j) for j in range(len(window))])
        nowcast[i] = np.sum(weights * window) / np.sum(weights)
    
    return nowcast

def calculate_nowcast(df, species_columns=None, avg_hours=None):
    """
    Calculate NowCast AQI for species in df, allowing custom column names.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with pollutant concentration columns
    species_columns : dict, optional
        Mapping of species names to df column names.
        Example: {'PM2.5': 'pm25_conc', 'NO2': 'no2_ppb', 'O3': 'ozone_ppb'}
        If None, defaults to species names as columns.
    avg_hours : dict, optional
        Averaging window hours per species (for NO2, O3).
        PM2.5 always uses NowCast method on raw hourly data.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added NowCast/AQI columns named:
        - PM2.5: 'PM25_NowCast_Concentration', 'PM25_NowCast_AQI'
        - Others: '{species}_AQI'
    """
    try:
        df = df.copy()
        if species_columns is None:
            species_columns = {'PM2.5': 'PM2.5', 'NO2': 'NO2', 'O3': 'O3'}
        if avg_hours is None:
            avg_hours = {}

        for species, col in species_columns.items():
            try:
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
            except (KeyError, ValueError, TypeError) as e:
                print(f"Error processing {species}: {e}")
                continue

        return df
    except Exception as e:
        print(f"Error in calculate_nowcast: {e}")
        return df if 'df' in locals() else pd.DataFrame()

def is_forecast_recent(file_path, hours_threshold=5):
    """
    Check if a forecast file was generated within the last N hours.
    
    Parameters
    ----------
    file_path : str
        Path to the forecast file
    hours_threshold : int
        Number of hours to check (default: 5)
        
    Returns
    -------
    bool
        True if file exists and was modified within the threshold, False otherwise
    """
    try:
        if not os.path.exists(file_path):
            return False
        
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        time_threshold = datetime.now() - timedelta(hours=hours_threshold)
        
        return file_mod_time > time_threshold
    except (OSError, ValueError, OverflowError) as e:
        print(f"Error checking forecast recency for {file_path}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error in is_forecast_recent: {e}")
        return False

def check_s3_access(bucket, prefix):
    """
    Check access to a specific S3 bucket and prefix.
    
    Parameters
    ----------
    bucket : str
        S3 bucket name
    prefix : str
        S3 prefix (folder path)
        
    Returns
    -------
    bool
        True if access is confirmed, False otherwise
    """
    s3 = boto3.client("s3")
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        if "Contents" in response:
            print(f"Access confirmed for s3://{bucket}/{prefix}")
            return True
        else:
            print(f"No files found, but access confirmed for s3://{bucket}/{prefix}")
            return True
    except ClientError as e:
        print(f" Access denied or error for s3://{bucket}/{prefix}: {e}")
        return False

def check_s3_connectivity(bucket, prefixes):
    """
    Check S3 connectivity for multiple prefixes.
    
    Parameters
    ----------
    bucket : str
        S3 bucket name
    prefixes : list
        List of S3 prefixes to check
    """
    for prefix in prefixes:
        check_s3_access(bucket, prefix)

def upload_to_s3(file_path, s3_client, s3_bucket):
    """Upload file to S3 bucket with verification.
    
    Parameters
    ----------
    file_path : str
        Local file path
    s3_client : boto3.client
        S3 client instance
    s3_bucket : str
        S3 bucket URI (s3:// prefix optional)
    """
    try:
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist. Skipping S3 upload.")
            return
        
        s3_bucket_name = s3_bucket.replace("s3://", "")
        s3_key = f"snwg_forecast_working_files/precomputed/all_dts/{os.path.basename(file_path)}"
        
        try:
            s3_client.upload_file(file_path, s3_bucket_name, s3_key)
            print(f"Successfully uploaded to s3://{s3_bucket_name}/{s3_key}")
        except (OSError, IOError) as e:
            print(f"File read error uploading {file_path}: {e}")
        except Exception as err:
            print(f"Failed to upload {file_path} to S3: {err}")
    except Exception as e:
        print(f"Unexpected error in upload_to_s3: {e}")