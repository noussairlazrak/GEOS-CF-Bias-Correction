import os
import json
from datetime import datetime, timedelta
import gzip
from timezonefinder import TimezoneFinder
import pytz

input_dirs = {
    'precomputed/all_dts': 'no2'
}
output_file = 'precomputed/combined_forecasts.json.gz'

now = datetime.now()
cutoff = now + timedelta(hours=48)

combined_data = []
tf = TimezoneFinder()


def parse_time(time_str):
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S'):
        try:
            return datetime.strptime(time_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Time data '{time_str}' does not match expected formats")




def process_directory(input_dir, species):
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, 'r') as f:
                try:
                    data = json.load(f)

                    location = data.get('location', 'Unknown')
                    fcst_species = data.get('species')
                    lat = data.get('lat')
                    lon = data.get('lon')
                    forecasts = data.get('forecasts', [])
                    tz_name = None
                    if lat is not None and lon is not None:
                        try:
                            tz_name = tf.timezone_at(lng=lon, lat=lat)
                        except Exception as e:
                            print(f"Warning: Could not determine timezone for {location} ({lat}, {lon}): {e}")

                    tz = pytz.timezone(tz_name) if tz_name else None

                    filtered_forecasts = []
                    for forecast in forecasts:
                        time_str = forecast.get('time')
                        if not time_str:
                            continue
                        try:
                            # Parse time supporting both formats
                            utc_dt = None
                            for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S'):
                                try:
                                    utc_dt = datetime.strptime(time_str, fmt)
                                    break
                                except ValueError:
                                    pass
                            if utc_dt is None:
                                raise ValueError(f"Time data '{time_str}' does not match expected formats")

                            if utc_dt > cutoff:
                                continue

                            local_time_str = None
                            if tz:
                                utc_dt = pytz.utc.localize(utc_dt)
                                local_dt = utc_dt.astimezone(tz)
                                local_time_str = local_dt.strftime('%Y-%m-%d %H:%M:%S')
                            forecast['local_time'] = local_time_str
                            filtered_forecasts.append(forecast)
                        except Exception as e:
                            print(f"Warning: Could not process forecast time in {filename}: {e}")


                    combined_data.append({
                        'location': location,
                        'lat': lat,
                        'lon': lon,
                        'timezone': tz_name,
                        'species': fcst_species,
                        'forecasts': filtered_forecasts
                    })

                except json.JSONDecodeError:
                    print(f"Warning: Could not parse {filename}")

for dir_path, species_name in input_dirs.items():
    process_directory(dir_path, species_name)

with gzip.open(output_file, 'wt', encoding='utf-8') as out_file:
    json.dump(combined_data, out_file, indent=2)

print(f"Saved combined forecasts to {output_file}")
