import sys
import os
import pickle
from datetime import datetime, timedelta
import datetime as dt
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pyod.models.iforest import IForest
import joblib 

# Local imports
sys.path.insert(1, 'MLpred')
from MLpred import mlpred
from MLpred import funcs


def get_localised_forecast(
    loc='', spec='no2', lat=0.0, lon=0.0, mod_src='s3', obs_src='pandora',
    openaq_id=None, GEOS_CF=None, OBS_URL=None, st=None, ed=None,
    resamp='1h', unit='ppb', interpol=True, rmv_out=True,
    time_col='time', date_fmt='%Y-%m-%d %H:%M', obs_val_col='unit',
    lat_col=None, lon_col=None, silent=False, force_retrain=True, **kwargs
):
    """Localized forecast: trains a basic LightGBM model on observations + historical GEOS-CF, 
    then predicts on V2 GEOS-CF forecasts.
    """
   
    if st is None: st = dt.datetime(2018, 1, 1)
    if ed is None: ed = dt.datetime.today()
    
    # Sanitize location name for file paths
    loc_clean = loc.replace(' ', '_').replace(',', '').lower()
    model_path = f"MODELS/lgbm_{loc_clean}_{spec}_basic.joblib"
    feature_path = f"MODELS/lgbm_{loc_clean}_{spec}_features_basic.pkl"
    model_exists = os.path.exists(model_path) and os.path.exists(feature_path) and not force_retrain
    
    try:
        # Load observations
        if not silent:
            print(f"INFO: Loading observations...")
        site = mlpred.ObsSite(openaq_id, model_source=mod_src, species=spec, observation_source=obs_src)
        site._silent = silent
        site.read_obs(source=obs_src, url=OBS_URL, time_col=time_col, date_format=date_fmt,
                      value_collum=obs_val_col, lat_col=lat_col, lon_col=lon_col, species=spec,
                      lat=lat, lon=lon, unit=unit, remove_outlier=rmv_out, **kwargs)
        
        if site._obs is None or (hasattr(site._obs, 'empty') and site._obs.empty):
            print(f"ERROR: No observations loaded.")
            return None, None, None
        
        obs_data = site._obs.copy()
        obs_data["time"] = pd.to_datetime(obs_data["time"]).dt.floor("H")
        
        if not silent:
            print(f"INFO: Observations loaded. Time range: {obs_data['time'].min()} to {obs_data['time'].max()}")
            print(f"INFO: {len(obs_data)} observation records")
        
        if obs_data.empty:
            print(f"ERROR: No observations loaded after processing")
            return None, None, None
        
        # Load GEOS-CF data (V1 for historical, V2 for recent/forecast)
        if not silent:
            print(f"INFO: Loading GEOS-CF V1 data (historical)...")
        
        geos_v1 = mlpred.read_geos_cf(lon=lon, lat=lat, start=st, end=ed, 
                                      version=1, verbose=not silent)
        
        if not silent:
            print(f"INFO: Loading GEOS-CF V2 data (recent + forecast)...")
        
        now = datetime.now()
        geos_v2 = mlpred.read_geos_cf(lon=lon, lat=lat, start=st, end=now + timedelta(days=5), 
                                      version=2, verbose=not silent)
        
        # Combine V1 and V2 data
        geos_data = None
        if geos_v1 is not None and not geos_v1.empty:
            geos_v1["time"] = pd.to_datetime(geos_v1["time"]).dt.floor("H")
            geos_data = geos_v1.copy()
            if not silent:
                print(f"INFO: GEOS-CF V1: {len(geos_v1)} samples ({geos_v1['time'].min()} to {geos_v1['time'].max()})")
        
        if geos_v2 is not None and not geos_v2.empty:
            geos_v2["time"] = pd.to_datetime(geos_v2["time"]).dt.floor("H")
            if geos_data is not None:
                # Combine, keeping V1 for overlapping times (more reliable historical data)
                geos_data = pd.concat([geos_data, geos_v2], ignore_index=True)
                geos_data = geos_data.drop_duplicates(subset=['time'], keep='first')
            else:
                geos_data = geos_v2.copy()
            if not silent:
                print(f"INFO: GEOS-CF V2: {len(geos_v2)} samples ({geos_v2['time'].min()} to {geos_v2['time'].max()})")
        
        if geos_data is None or geos_data.empty:
            print(f"ERROR: No GEOS-CF data retrieved")
            return None, None, None
        
        geos_data = geos_data.sort_values('time').reset_index(drop=True)
        
        if not silent:
            print(f"INFO: Combined GEOS-CF data: {len(geos_data)} samples")
            print(f"INFO: Time range: {geos_data['time'].min()} to {geos_data['time'].max()}")
        
        # Train model if doesn't exist
        if not model_exists:
            if not silent:
                print(f"INFO: Training basic model...")
            
            train_data = geos_data.copy()
            
            # Merge observations with GEOS-CF
            merged_train = train_data.merge(obs_data[["time", "value"]], on="time", how="inner")
            merged_train = merged_train.dropna(subset=["value"])
            
            if not silent:
                print(f"INFO: Training data: {len(merged_train)} samples")
            
            if len(merged_train) < 50:
                print(f"ERROR: Insufficient training data ({len(merged_train)} samples)")
                return None, None, None
            
            # Simple feature selection: use all numeric columns
            yvar = "value"
            skipvar = ["time", "location", "lat", "lon", yvar]
            
            numeric_cols = merged_train.select_dtypes(include=[np.number]).columns.tolist()
            sel_feats = [c for c in numeric_cols if c not in skipvar]
            
            if not silent:
                print(f"INFO: Using {len(sel_feats)} features")
            
            # Remove outliers
            if rmv_out:
                conc_obs = merged_train[yvar].values.reshape(-1, 1)
                model_IF = IForest(contamination=0.02)
                model_IF.fit(conc_obs)
                anomalies = model_IF.predict(conc_obs)
                n_removed = (anomalies == 1).sum()
                merged_train = merged_train[anomalies != 1]
                if not silent:
                    print(f"INFO: Removed {n_removed} outliers, {len(merged_train)} samples remaining")
            
            # Simple 80/20 time-based split
            merged_train_sorted = merged_train.sort_values('time')
            split_idx = int(len(merged_train_sorted) * 0.8)
            
            tx = merged_train_sorted[sel_feats].iloc[:split_idx]
            ty = merged_train_sorted[yvar].iloc[:split_idx]
            vx = merged_train_sorted[sel_feats].iloc[split_idx:]
            vy = merged_train_sorted[yvar].iloc[split_idx:]
            
            tx, ty = funcs.clean_data(tx, ty)
            vx, vy = funcs.clean_data(vx, vy)
            tx = funcs.clean_feature_names(tx)
            vx = funcs.clean_feature_names(vx)
            
            if not silent:
                print(f"INFO: Training set: {len(tx)}, Validation set: {len(vx)}")
            
            # Basic LightGBM model with simple parameters
            model_lgb = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                num_leaves=31,
                verbosity=-1,
                random_state=42
            )
            model_lgb.fit(tx, ty)
            
            # Calculate metrics
            vy_pred = model_lgb.predict(vx)
            ty_pred = model_lgb.predict(tx)
            
            rmse = round(np.sqrt(mean_squared_error(vy, vy_pred)), 2)
            r2 = round(r2_score(vy, vy_pred), 2)
            mae = round(mean_absolute_error(vy, vy_pred), 2)
            train_r2 = round(r2_score(ty, ty_pred), 2)
            
            metrics = {'RMSE': rmse, 'R2': r2, 'MAE': mae, 'Train_R2': train_r2}
            
            if not silent:
                print(f"INFO: Model trained.")
                print(f"      Training R2: {train_r2}")
                print(f"      Validation RMSE={rmse}, R2={r2}, MAE={mae}")
            
            # Save model
            try:
                os.makedirs("MODELS", exist_ok=True)
                joblib.dump(model_lgb, model_path)
                pickle.dump(sel_feats, open(feature_path, 'wb'))
                if not silent:
                    print(f"INFO: Model saved to {model_path}")
            except Exception as e:
                if not silent:
                    print(f"WARNING: Could not save model: {e}")
        else:
            if not silent:
                print(f"INFO: Loading pretrained model from {model_path}")
            model_lgb = joblib.load(model_path)
            sel_feats = pickle.load(open(feature_path, 'rb'))
            metrics = {'RMSE': 0, 'R2': 0, 'MAE': 0}
        
        # Use combined GEOS-CF V1+V2 data for predictions
        if not silent:
            print(f"INFO: Preparing combined GEOS-CF data for predictions...")
        
        all_geos_data = geos_data.copy()
        all_geos_data = all_geos_data.sort_values('time').reset_index(drop=True)
        
        # Check for gaps in the time series
        time_diff = all_geos_data['time'].diff()
        expected_diff = pd.Timedelta(hours=1)
        gaps = time_diff[time_diff > expected_diff * 2]  # Gaps larger than 2 hours
        if len(gaps) > 0 and not silent:
            print(f"WARNING: Found {len(gaps)} gaps in GEOS-CF time series")
            for idx in gaps.index[:5]:  # Show first 5 gaps
                gap_start = all_geos_data.loc[idx-1, 'time'] if idx > 0 else 'N/A'
                gap_end = all_geos_data.loc[idx, 'time']
                print(f"         Gap: {gap_start} to {gap_end} ({time_diff.loc[idx]})")
        
        if not silent:
            print(f"INFO: Full GEOS-CF (V1+V2) data: {len(all_geos_data)} samples")
            print(f"INFO: Time range: {all_geos_data['time'].min()} to {all_geos_data['time'].max()}")
        
        # Make predictions on ALL GEOS-CF data
        if not silent:
            print(f"INFO: Making predictions for entire GEOS-CF period...")
        
        # Clean feature names in the prediction data
        all_geos_data = funcs.clean_feature_names(all_geos_data)
        
        # Clean the selected feature names the same way
        sel_feats_clean = []
        for f in sel_feats:
            temp_df = pd.DataFrame(columns=[f])
            temp_df = funcs.clean_feature_names(temp_df)
            sel_feats_clean.append(temp_df.columns[0])
        
        if not silent:
            print(f"DEBUG: Model features (first 5): {sel_feats_clean[:5]}")
            print(f"DEBUG: Available columns (first 10): {list(all_geos_data.columns)[:10]}")
        
        sel_feats_avail = [f for f in sel_feats_clean if f in all_geos_data.columns]
        
        if not silent:
            print(f"INFO: {len(sel_feats_avail)}/{len(sel_feats_clean)} features available for prediction")
        
        if not sel_feats_avail:
            print(f"ERROR: None of the model features found in GEOS-CF data")
            print(f"DEBUG: Looking for: {sel_feats_clean[:10]}")
            print(f"DEBUG: Available: {list(all_geos_data.columns)[:20]}")
            return None, None, None
        
        all_geos_data["localised"] = np.nan
        
        # Check which rows have valid data for all features
        valid_mask = all_geos_data[sel_feats_avail].notnull().all(axis=1)
        n_valid = valid_mask.sum()
        
        if not silent:
            print(f"DEBUG: {n_valid} rows have all features available")
            if n_valid == 0:
                # Check each feature for nulls
                for feat in sel_feats_avail[:10]:
                    n_null = all_geos_data[feat].isna().sum()
                    print(f"DEBUG: Feature '{feat}' has {n_null} null values out of {len(all_geos_data)}")
        
        if n_valid > 0:
            X_pred = all_geos_data.loc[valid_mask, sel_feats_avail].copy()
            
            # Get model's expected features
            model_features = list(model_lgb.feature_name_)
            
            if not silent:
                print(f"DEBUG: Model expects {len(model_features)} features")
                missing_in_pred = [f for f in model_features if f not in X_pred.columns]
                if missing_in_pred:
                    print(f"DEBUG: Missing features in prediction data: {missing_in_pred[:5]}")
            
            # Add missing features with 0 values
            for feat in model_features:
                if feat not in X_pred.columns:
                    X_pred[feat] = 0.0
            
            # Reorder to match model's expected order
            X_pred = X_pred[model_features]
            
            predictions = model_lgb.predict(X_pred)
            all_geos_data.loc[valid_mask, "localised"] = predictions
            
            if not silent:
                print(f"INFO: Generated {n_valid} predictions")
                print(f"INFO: Prediction range: {predictions.min():.2f} to {predictions.max():.2f}")
        else:
            print(f"ERROR: No valid rows for prediction (all rows have missing features)")
        
        # Interpolate small gaps in localised predictions (up to 6 hours)
        n_missing_before = all_geos_data["localised"].isna().sum()
        if n_missing_before > 0:
            all_geos_data["localised"] = all_geos_data["localised"].interpolate(method='linear', limit=6)
            n_missing_after = all_geos_data["localised"].isna().sum()
            if not silent and n_missing_before != n_missing_after:
                print(f"INFO: Interpolated {n_missing_before - n_missing_after} missing values in localised predictions")
        
        # Merge observations (where available) with full GEOS-CF predictions
        result = all_geos_data.merge(obs_data[["time", "value"]], on="time", how="left")
        
        if not silent:
            n_with_obs = result["value"].notna().sum()
            print(f"INFO: Result shape: {result.shape}")
            print(f"INFO: {n_with_obs} timestamps have observations, {len(result) - n_with_obs} are predictions only")
        
        return result, metrics, model_lgb
        
    except Exception as e:
        print(f"ERROR: Forecast generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


site_settings = {'loc': 'ManhattanNY-CCNY',
                 'spec': 'no2',
                 'lat': 40.8153,
                 'lon': -73.9505,
                 'silent': False,
                 'mod_src': 's3',
                 'obs_src': 'pandora',
                 'openaq_id': None,
                 'GEOS_CF': '#',
                 'OBS_URL': 'https://data.pandonia-global-network.org/ManhattanNY-CCNY//Pandora135s1//L2/Pandora135s1_ManhattanNY-CCNY_L2_rnvh3p1-8.txt',
                 'resamp' : '1h',
                 'unit' : 'ppbv',
                 'interpol': False,
                 'rmv_out': False,
                 'st' : dt.datetime(2018, 1, 1),
                 'ed': dt.datetime.today()
                }
obs_settings = {'time_col': 'time',
                'date_fmt': '%Y-%m-%d %H:%M',
                'obs_val_col': 'unit',
                'lat_col': None,
                'lon_col': None,
               }
merged_data, metrics, model = get_localised_forecast(
    **site_settings, **obs_settings
)