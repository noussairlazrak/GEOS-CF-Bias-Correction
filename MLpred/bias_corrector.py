# -*- coding: utf-8 -*-
"""
bias_corrector.py

Localised bias-correction forecasts using LightGBM trained on
Pandora (or other) observations merged with GEOS-CF V1 (historical replay)
and V2 (recent analysis + 5-day forecast).

Key features
------------
* Observations are loaded via ``read_pandora`` with local caching so that
  repeated calls for the same site do not re-download the full file.
* Pre-trained models are looked up first from the local ``MODELS/`` folder,
  then from S3 (``snwg_forecast_working_files/MODELS/``), so forecast
  generation is fast after the first run.
* GEOS-CF V1 (replay) is used for training; V2 provides the most-recent
  analysis window and the 5-day forecast horizon.

Author: Noussair Lazrak
"""

import sys
import os
import pickle
from datetime import datetime, timedelta
import datetime as dt
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from pyod.models.iforest import IForest
import joblib

try:
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    _cwd = os.path.abspath(os.getcwd())
    _REPO_ROOT = _cwd
    for _candidate in [_cwd, os.path.dirname(_cwd)]:
        if os.path.isdir(os.path.join(_candidate, "MLpred")):
            _REPO_ROOT = _candidate
            break

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from MLpred import mlpred
from MLpred import funcs
from MLpred.pandora import read_pandora, _cache_path, _is_cache_fresh, DEFAULT_CACHE_HOURS
from MLpred.s3_manager import S3Manager

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS_LOCAL_DIR = os.path.join(_REPO_ROOT, "MODELS")
S3_BUCKET        = os.environ.get("GEOS_CF_S3_BUCKET", "smce-geos-cf-public")
S3_MODELS_PREFIX = "snwg_forecast_working_files/MODELS"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _model_local_paths(loc_clean: str, spec: str):
    """Return (model_path, feature_path, metrics_path) for a given location + species."""
    model_path   = os.path.join(MODELS_LOCAL_DIR, f"lgbm_{loc_clean}_{spec}_basic.joblib")
    feature_path = os.path.join(MODELS_LOCAL_DIR, f"lgbm_{loc_clean}_{spec}_features_basic.pkl")
    metrics_path = os.path.join(MODELS_LOCAL_DIR, f"lgbm_{loc_clean}_{spec}_metrics.json")
    return model_path, feature_path, metrics_path


def _is_model_fresh(loc_clean: str, spec: str, max_age_days: int = 7) -> bool:
    """Return True if the local model files exist and are younger than *max_age_days*."""
    model_path, feature_path, _ = _model_local_paths(loc_clean, spec)
    if not (os.path.exists(model_path) and os.path.exists(feature_path)):
        return False
    age_seconds = dt.datetime.now().timestamp() - os.path.getmtime(model_path)
    return age_seconds < max_age_days * 86400


def _try_load_model_from_s3(loc_clean: str, spec: str,
                             s3_manager: S3Manager,
                             silent: bool = False):
    """
    Try to download a pre-trained model from S3 into the local MODELS/ folder.
    Returns (model, payload, persisted_metrics) if successful, (None, None, None) otherwise.
    """
    model_path, feature_path, metrics_path = _model_local_paths(loc_clean, spec)
    model_s3_key   = f"{S3_MODELS_PREFIX}/lgbm_{loc_clean}_{spec}_basic.joblib"
    feature_s3_key = f"{S3_MODELS_PREFIX}/lgbm_{loc_clean}_{spec}_features_basic.pkl"
    metrics_s3_key = f"{S3_MODELS_PREFIX}/lgbm_{loc_clean}_{spec}_metrics.json"

    os.makedirs(MODELS_LOCAL_DIR, exist_ok=True)

    try:
        got_model = s3_manager.download_file(model_s3_key,   model_path)
        got_feats = s3_manager.download_file(feature_s3_key, feature_path)

        if got_model and got_feats:
            model   = joblib.load(model_path)
            payload = pickle.load(open(feature_path, "rb"))
            # Try to load persisted metrics (best-effort)
            persisted_metrics = None
            try:
                s3_manager.download_file(metrics_s3_key, metrics_path)
                if os.path.exists(metrics_path):
                    import json as _json
                    with open(metrics_path) as _f:
                        persisted_metrics = _json.load(_f)
            except Exception:
                pass
            if not silent:
                print(f"INFO: Loaded pre-trained model from S3 → {model_s3_key}")
            return model, payload, persisted_metrics
    except Exception as exc:
        if not silent:
            print(f"INFO: S3 model not available ({exc})")

    return None, None, None


def _save_model(model, features, loc_clean: str, spec: str,
                s3_manager: S3Manager, silent: bool = False,
                target_mode: str = "absolute", metrics: dict = None):
    """Persist model + feature list + metrics locally and upload to S3."""
    model_path, feature_path, metrics_path = _model_local_paths(loc_clean, spec)
    os.makedirs(MODELS_LOCAL_DIR, exist_ok=True)

    try:
        joblib.dump(model, model_path)
        payload = {"features": features, "target": target_mode}
        pickle.dump(payload, open(feature_path, "wb"))
        if metrics:
            import json as _json
            with open(metrics_path, "w") as _f:
                _json.dump(metrics, _f)
        if not silent:
            print(f"INFO: Model saved locally → {model_path}")
    except Exception as exc:
        if not silent:
            print(f"WARNING: Could not save model locally: {exc}")

    try:
        model_s3_key   = f"{S3_MODELS_PREFIX}/lgbm_{loc_clean}_{spec}_basic.joblib"
        feature_s3_key = f"{S3_MODELS_PREFIX}/lgbm_{loc_clean}_{spec}_features_basic.pkl"
        metrics_s3_key = f"{S3_MODELS_PREFIX}/lgbm_{loc_clean}_{spec}_metrics.json"
        s3_manager.upload_file(model_path,   model_s3_key)
        s3_manager.upload_file(feature_path, feature_s3_key)
        if metrics and os.path.exists(metrics_path):
            s3_manager.upload_file(metrics_path, metrics_s3_key)
        if not silent:
            print(f"INFO: Model uploaded to S3 → {model_s3_key}")
    except Exception as exc:
        if not silent:
            print(f"WARNING: Could not upload model to S3: {exc}")


def _load_observations(obs_src: str, obs_url: str,
                       openaq_id, spec: str,
                       lat: float, lon: float,
                       mod_src: str, unit: str,
                       time_col: str, date_fmt: str,
                       obs_val_col: str, lat_col, lon_col,
                       rmv_out: bool, cache_hours: int,
                       silent: bool, force_refresh: bool = False,
                       **kwargs) -> pd.DataFrame:
    """
    Load observations, using ``read_pandora`` 
    sources and the standard ``mlpred.ObsSite`` for other sources 
    """
    if obs_src == "pandora" and obs_url:
        cache_file = _cache_path(spec, url=obs_url)
        if force_refresh and os.path.exists(cache_file):
            os.remove(cache_file)
            if not silent:
                print(f"INFO: Pandora cache deleted (force_obs_refresh) → {os.path.basename(cache_file)}")
        if not silent:
            fresh = _is_cache_fresh(cache_file, cache_hours)
            print(f"INFO: Loading Pandora obs — cache {'hit' if fresh else 'miss'} "
                  f"({os.path.basename(cache_file)})")
        df = read_pandora(obs_url, pollutant=spec,
                          cache=True, cache_hours=cache_hours, silent=silent)
        df["time"] = pd.to_datetime(df["time"]).dt.floor("H")
        return df

    # Generic path (OpenAQ, local CSV, …)
    site = mlpred.ObsSite(openaq_id, model_source=mod_src,
                          species=spec, observation_source=obs_src)
    site._silent = silent
    site.read_obs(
        source=obs_src, url=obs_url, time_col=time_col,
        date_format=date_fmt, value_collum=obs_val_col,
        lat_col=lat_col, lon_col=lon_col, species=spec,
        lat=lat, lon=lon, unit=unit, remove_outlier=rmv_out, **kwargs,
    )
    if site._obs is None or (hasattr(site._obs, "empty") and site._obs.empty):
        return pd.DataFrame()
    df = site._obs.copy()
    df["time"] = pd.to_datetime(df["time"]).dt.floor("H")
    return df


def _add_atmospheric_features(df: pd.DataFrame, spec: str = "no2") -> pd.DataFrame:
    """
    Enrich a GEOS-CF (or merged) DataFrame with engineered features that
    capture atmospheric and temporal patterns the raw model variables miss.

    Added feature groups
    --------------------
    Cyclical time
        ``hour_sin``, ``hour_cos``  – diurnal cycle (avoids 0/23 discontinuity)
        ``doy_sin``,  ``doy_cos``   – annual cycle (avoids 1/365 discontinuity)
        ``weekday``                 – integer 0-6 (Mon-Sun)
        ``is_weekend``              – binary flag

    Atmospheric proxies
        ``pbl_proxy``  – log(PBLH) if a PBL-height column is present; captures
                         mixing-layer dilution effect on surface concentrations.
        ``t2m_delta``  – T2M − T850 (lapse-rate proxy for convective mixing)
                         only added when both columns exist.

    Lagged species
        ``{spec}_lag1h``, ``{spec}_lag3h``, ``{spec}_lag6h``
        ``{spec}_lag24h`` – yesterday-same-hour persistence signal
        ``{spec}_roll3h``,``{spec}_roll6h`` – short rolling means for trend context

    """
    out = df.copy()

    # Cyclical 
    t = pd.to_datetime(out["time"])
    out["hour_sin"]  = np.sin(2 * np.pi * t.dt.hour / 24)
    out["hour_cos"]  = np.cos(2 * np.pi * t.dt.hour / 24)
    out["doy_sin"]   = np.sin(2 * np.pi * t.dt.dayofyear / 365.25)
    out["doy_cos"]   = np.cos(2 * np.pi * t.dt.dayofyear / 365.25)
    out["weekday"]   = t.dt.weekday.astype(float)
    out["is_weekend"] = (t.dt.weekday >= 5).astype(float)

    # Atmospheric proxy 
    pbl_candidates = [c for c in out.columns
                      if "pbl" in c.lower() or "pbh" in c.lower() or "kpbl" in c.lower()]
    if pbl_candidates:
        pbl_col = pbl_candidates[0]
        out["pbl_proxy"] = np.log1p(out[pbl_col].clip(lower=0))

    # surface–850 hPa temperature difference
    t2m_cols  = [c for c in out.columns if "t2m"  in c.lower() or "ts"   in c.lower()]
    t850_cols = [c for c in out.columns if "t850" in c.lower() or "t_850" in c.lower()]
    if t2m_cols and t850_cols:
        out["t2m_delta"] = out[t2m_cols[0]] - out[t850_cols[0]]

    # Lagged 
    spec_col = None
    for candidate in [spec, f"{spec}_col", f"geos_{spec}"]:
        if candidate in out.columns:
            spec_col = candidate
            break

    if spec_col is None:
        hits = [c for c in out.columns if spec.lower() in c.lower() and c != "value"]
        if hits:
            spec_col = hits[0]

    if spec_col is not None:
        s = out[spec_col]
        out[f"{spec}_lag1h"]   = s.shift(1)
        out[f"{spec}_lag3h"]   = s.shift(3)
        out[f"{spec}_lag6h"]   = s.shift(6)
        out[f"{spec}_lag24h"]  = s.shift(24)
        out[f"{spec}_roll3h"]  = s.rolling(3,  min_periods=1).mean()
        out[f"{spec}_roll6h"]  = s.rolling(6,  min_periods=1).mean()

    # Fill NaN lags with column medians
    new_cols = [c for c in out.columns if c not in df.columns]
    for c in new_cols:
        med = out[c].median()
        if pd.isna(med):
            med = 0.0
        out[c] = out[c].fillna(med)

    return out

def get_localised_forecast(
    loc: str = "",
    spec: str = "no2",
    lat: float = 0.0,
    lon: float = 0.0,
    mod_src: str = "s3",
    obs_src: str = "pandora",
    openaq_id=None,
    GEOS_CF=None,
    OBS_URL: str = None,
    st=None,
    ed=None,
    resamp: str = "1h",
    unit: str = "ppb",
    interpol: bool = True,
    rmv_out: bool = True,
    time_col: str = "time",
    date_fmt: str = "%Y-%m-%d %H:%M",
    obs_val_col: str = "unit",
    lat_col=None,
    lon_col=None,
    silent: bool = False,
    force_retrain: bool = False,
    force_obs_refresh: bool = False,
    model_max_age_days: int = 7,
    obs_cache_hours: int = DEFAULT_CACHE_HOURS,
    s3_manager: S3Manager = None,
    **kwargs,
):
    """
    Generate a localised, bias-corrected NO2 / O3 / PM2.5 forecast.

    Workflow
    --------
    1. **Observations** – loaded via ``read_pandora`` (Pandora sources) or
       ``mlpred.ObsSite`` (other sources).  Pandora observations are cached
       locally in ``OBS/<Site>_<spec>.csv`` and re-used when fresh.

    2. **GEOS-CF data**
       * *V1 (replay)* – full historical record used for training.
       * *V2 (analysis + forecast)* – recent data + 5-day ahead forecast.
         V2 fills gaps in V1 and extends the time series into the future.

    3. **Model lookup order**
       a. Local ``MODELS/`` folder.
       b. S3 bucket (``snwg_forecast_working_files/MODELS/``).
       c. Train a new LightGBM model if none is found (or ``force_retrain``).

    4. **Prediction** – the trained model predicts over the entire combined
       V1 + V2 GEOS-CF time series.  Small gaps (≤ 6 h) are linearly
       interpolated.

    Parameters
    ----------
    loc : str
        Human-readable location name (used for model file names).
    spec : {'no2', 'o3', 'pm25'}
        Target species.
    lat, lon : float
        Site coordinates.
    mod_src : str
        GEOS-CF data source (``'s3'`` or ``'local'``).
    obs_src : str
        Observation source (``'pandora'``, ``'openaq'``, ``'local'``, …).
    OBS_URL : str
        URL or file path of the observation file.
    st, ed : datetime
        Training date range.
    force_retrain : bool
        Re-train even if a saved model is found (default: ``False``).
    force_obs_refresh : bool
        Delete and re-download the Pandora cache even if it is fresh
        (default: ``False``).  Use this when you suspect the cached obs
        file is incomplete or covers too narrow a date range.
    obs_cache_hours : int
        Pandora cache staleness threshold in hours (default: ``DEFAULT_CACHE_HOURS``).
    s3_manager : S3Manager, optional
        Pre-configured S3Manager.  Created automatically from ``S3_BUCKET``
        if not supplied.

    Returns
    -------
    result : pd.DataFrame or None
    metrics : dict or None
    model : LGBMRegressor or None
    """
    if st is None:
        st = dt.datetime(2018, 1, 1)
    if ed is None:
        ed = dt.datetime.today()
    if s3_manager is None:
        s3_manager = S3Manager(bucket_name=S3_BUCKET)

    loc_clean  = loc.replace(" ", "_").replace(",", "").lower()
    model_path, feature_path, metrics_path = _model_local_paths(loc_clean, spec)

    try:
        # 1. Observations ──────────────────────────────────────────────────
        if not silent:
            print(f"INFO: Loading observations for {loc} ({obs_src})…")

        obs_data = _load_observations(
            obs_src=obs_src, obs_url=OBS_URL, openaq_id=openaq_id,
            spec=spec, lat=lat, lon=lon, mod_src=mod_src, unit=unit,
            time_col=time_col, date_fmt=date_fmt, obs_val_col=obs_val_col,
            lat_col=lat_col, lon_col=lon_col, rmv_out=rmv_out,
            cache_hours=obs_cache_hours, force_refresh=force_obs_refresh,
            silent=silent, **kwargs,
        )

        if obs_data is None or obs_data.empty:
            print(f"ERROR: No observations loaded for {loc}")
            return None, None, None

        if not silent:
            print(f"INFO: {len(obs_data)} observations "
                  f"({obs_data['time'].min()} → {obs_data['time'].max()})")

        # 2. GEOS-CF V1 (historical replay) ───────────────────────────────
        if not silent:
            print(f"INFO: Loading GEOS-CF V1 (replay)…")
        geos_v1 = mlpred.read_geos_cf(lon=lon, lat=lat, start=st, end=ed,
                                       version=1, verbose=not silent)

        # 3. GEOS-CF V2 (recent analysis + 5-day forecast) ────────────────
        if not silent:
            print(f"INFO: Loading GEOS-CF V2 (analysis + forecast)…")
        geos_v2 = mlpred.read_geos_cf(
            lon=lon, lat=lat,
            start=st,
            end=datetime.now() + timedelta(days=5),
            version=2, verbose=not silent,
        )

        # 4. Combine V1 + V2 (V1 wins on overlapping timestamps) ──────────
        frames = []
        if geos_v1 is not None and not geos_v1.empty:
            geos_v1["time"] = pd.to_datetime(geos_v1["time"]).dt.floor("H")
            frames.append(geos_v1)
            if not silent:
                print(f"INFO: V1 → {len(geos_v1)} rows "
                      f"({geos_v1['time'].min()} → {geos_v1['time'].max()})")

        if geos_v2 is not None and not geos_v2.empty:
            geos_v2["time"] = pd.to_datetime(geos_v2["time"]).dt.floor("H")
            frames.append(geos_v2)
            if not silent:
                print(f"INFO: V2 → {len(geos_v2)} rows "
                      f"({geos_v2['time'].min()} → {geos_v2['time'].max()})")

        if not frames:
            print(f"ERROR: No GEOS-CF data available for {loc}")
            return None, None, None

        geos_data = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=["time"], keep="first")   # V1 first = V1 wins
            .sort_values("time")
            .reset_index(drop=True)
        )
        if not silent:
            print(f"INFO: Combined GEOS-CF → {len(geos_data)} rows "
                  f"({geos_data['time'].min()} → {geos_data['time'].max()})")

        # 5. Model: local → S3 → train ────────────────────────────────────
        model_lgb, sel_feats, metrics = None, None, {}
        model_path, feature_path, metrics_path = _model_local_paths(loc_clean, spec)

        local_ready = (
            os.path.exists(model_path) and
            os.path.exists(feature_path) and
            not force_retrain and
            _is_model_fresh(loc_clean, spec, max_age_days=model_max_age_days)
        )

        if local_ready:
            if not silent:
                age_h = (dt.datetime.now().timestamp() - os.path.getmtime(model_path)) / 3600
                print(f"INFO: Loading model from local cache ({age_h:.1f}h old) → {model_path}")
            model_lgb = joblib.load(model_path)
            payload   = pickle.load(open(feature_path, "rb"))
            # Support both old format (plain list) and new format (dict)
            if isinstance(payload, dict):
                sel_feats   = payload["features"]
                target_mode_loaded = payload.get("target", "absolute")
            else:
                sel_feats   = payload
                target_mode_loaded = "absolute"
            metrics = {"source": "local", "target": target_mode_loaded}
            # Load persisted metrics if available
            if os.path.exists(metrics_path):
                try:
                    import json as _json
                    with open(metrics_path) as _f:
                        persisted = _json.load(_f)
                    metrics.update({k: persisted.get(k) for k in ("RMSE", "R2", "MAE", "n_train", "trained_at")})
                except Exception:
                    pass

        elif not force_retrain:
            if not silent:
                print(f"INFO: Local model not found — checking S3…")
            model_lgb, payload, persisted_metrics = _try_load_model_from_s3(
                loc_clean, spec, s3_manager, silent=silent
            )
            if model_lgb is not None:
                if isinstance(payload, dict):
                    sel_feats          = payload["features"]
                    target_mode_loaded = payload.get("target", "absolute")
                else:
                    sel_feats          = payload
                    target_mode_loaded = "absolute"
                metrics = {"source": "s3", "target": target_mode_loaded}
                if persisted_metrics:
                    metrics.update({k: persisted_metrics.get(k) for k in ("RMSE", "R2", "MAE", "n_train", "trained_at")})

        # Evaluate pre-loaded model on available obs/GEOS overlap
        if model_lgb is not None and metrics.get("RMSE") is None:
            try:
                merged_eval = (
                    geos_data.merge(obs_data[["time", "value"]], on="time", how="inner")
                             .dropna(subset=["value"])
                )
                if len(merged_eval) >= 10:
                    # Apply same feature engineering as training
                    merged_eval = _add_atmospheric_features(merged_eval, spec=spec)
                    eval_df = funcs.clean_feature_names(merged_eval.copy())

                    def _clean_feat(name: str) -> str:
                        tmp = pd.DataFrame(columns=[name])
                        return funcs.clean_feature_names(tmp).columns[0]

                    feats_clean = [_clean_feat(f) for f in sel_feats]
                    feats_avail = [f for f in feats_clean if f in eval_df.columns]
                    eval_X = eval_df[feats_avail].ffill().bfill().fillna(eval_df[feats_avail].median())
                    model_features = list(model_lgb.feature_name_)
                    for feat in model_features:
                        if feat not in eval_X.columns:
                            eval_X[feat] = 0.0
                    eval_X = eval_X[model_features]
                    eval_y = merged_eval["value"].values

                    raw_col_clean = _clean_feat(spec)
                    if metrics.get("target") == "ratio" and raw_col_clean in eval_df.columns:
                        # model predicts ratio; localised = raw * ratio
                        raw_vals   = eval_df[raw_col_clean].values
                        ratios     = model_lgb.predict(eval_X)
                        eval_preds = raw_vals * np.clip(ratios, 0.1, 10.0)
                    else:
                        eval_preds = model_lgb.predict(eval_X)

                    metrics.update({
                        "RMSE": round(np.sqrt(mean_squared_error(eval_y, eval_preds)), 2),
                        "R2":   round(r2_score(eval_y, eval_preds), 2),
                        "MAE":  round(mean_absolute_error(eval_y, eval_preds), 2),
                    })
                    if not silent:
                        print(f"INFO: Eval metrics (n={len(merged_eval)}) — "
                              f"RMSE={metrics['RMSE']}  R2={metrics['R2']}  MAE={metrics['MAE']}")
                else:
                    metrics.update({"RMSE": None, "R2": None, "MAE": None})
                    if not silent:
                        print(f"INFO: Too few obs/GEOS overlap rows ({len(merged_eval)}) to compute metrics")
            except Exception as exc:
                metrics.update({"RMSE": None, "R2": None, "MAE": None})
                if not silent:
                    print(f"WARNING: Could not compute eval metrics: {exc}")

        # Train if still no model
        if model_lgb is None:
            if not silent:
                print(f"INFO: Training new model for {loc}…")

            merged_train = (
                geos_data.merge(obs_data[["time", "value"]], on="time", how="inner")
                         .dropna(subset=["value"])
            )

            if not silent:
                print(f"INFO: Training samples after merge: {len(merged_train)}")

            if len(merged_train) < 50:
                print(f"ERROR: Insufficient training data ({len(merged_train)} samples)")
                return None, None, None

            # Features
            merged_train = _add_atmospheric_features(merged_train, spec=spec)
            if not silent:
                new_feat_count = len([c for c in merged_train.columns
                                      if c not in geos_data.columns and c != "value"])
                print(f"INFO: {new_feat_count} engineered features added "
                      f"(cyclical time, atmospheric proxies, lags)")

            # obs / geos_raw
        
            raw_col = spec 
            if raw_col not in merged_train.columns:

                hits = [c for c in merged_train.columns
                        if spec.lower() in c.lower() and c != "value"]
                if hits:
                    raw_col = hits[0]

            if raw_col in merged_train.columns:
                raw_vals = merged_train[raw_col].replace(0, np.nan)
                merged_train["bias_ratio"] = (merged_train["value"] / raw_vals).clip(0.05, 20.0)
                merged_train = merged_train.dropna(subset=["bias_ratio"])
                yvar = "bias_ratio"
                target_mode = "ratio"
                if not silent:
                    print(f"INFO: Target mode = bias ratio (obs / {raw_col})  "
                          f"median ratio = {merged_train['bias_ratio'].median():.3f}")
            else:

                yvar = "value"
                target_mode = "absolute"
                if not silent:
                    print(f"WARNING: Raw GEOS-CF column '{spec}' not found — "
                          f"falling back to absolute prediction")

            skip      = {"time", "location", "lat", "lon", "value", "bias_ratio"}
            num_cols  = merged_train.select_dtypes(include=[np.number]).columns.tolist()
            sel_feats = [c for c in num_cols if c not in skip]

            if rmv_out:
                conc = merged_train[yvar].values.reshape(-1, 1)
                clf  = IForest(contamination=0.02)
                clf.fit(conc)
                mask = clf.predict(conc) != 1
                if not silent:
                    print(f"INFO: Outlier removal: {(~mask).sum()} removed, "
                          f"{mask.sum()} remaining")
                merged_train = merged_train[mask]

            merged_train = merged_train.sort_values("time").reset_index(drop=True)

            # cross-validation
            n_splits = min(5, max(2, len(merged_train) // 500))
            tscv     = TimeSeriesSplit(n_splits=n_splits)
            fold_rmses, fold_r2s, fold_maes = [], [], []

            X_all_raw, y_all_raw = funcs.clean_data(merged_train[sel_feats],
                                                     merged_train[yvar])
            X_all = funcs.clean_feature_names(X_all_raw)
            y_all = y_all_raw.values

            if not silent:
                print(f"INFO: Time-series CV with {n_splits} folds "
                      f"({len(merged_train)} samples, target={yvar})…")

            for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_all), 1):
                tx, ty = X_all.iloc[tr_idx], y_all[tr_idx]
                vx, vy = X_all.iloc[val_idx], y_all[val_idx]

                _m = lgb.LGBMRegressor(
                    n_estimators=500, max_depth=5,
                    learning_rate=0.03, num_leaves=31,
                    subsample=0.8, colsample_bytree=0.8,
                    min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
                    verbosity=-1, random_state=42,
                )
                _m.fit(tx, ty,
                       eval_set=[(vx, vy)],
                       callbacks=[lgb.early_stopping(50, verbose=False),
                                  lgb.log_evaluation(-1)])

                pv = _m.predict(vx)
                # Compute metrics in observation space regardless of target mode
                if target_mode == "ratio" and raw_col in merged_train.columns:
                    raw_v = merged_train[raw_col].iloc[val_idx].values
                    vy_obs = merged_train["value"].iloc[val_idx].values
                    pv_obs = raw_v * np.clip(pv, 0.05, 20.0)
                else:
                    vy_obs, pv_obs = vy, pv

                fold_rmses.append(np.sqrt(mean_squared_error(vy_obs, pv_obs)))
                fold_r2s.append(r2_score(vy_obs, pv_obs))
                fold_maes.append(mean_absolute_error(vy_obs, pv_obs))
                if not silent:
                    print(f"  Fold {fold}/{n_splits}: "
                          f"RMSE={fold_rmses[-1]:.3f}  "
                          f"R²={fold_r2s[-1]:.3f}  "
                          f"MAE={fold_maes[-1]:.3f}  "
                          f"(val n={len(val_idx)})")

            cv_rmse = round(float(np.mean(fold_rmses)), 3)
            cv_r2   = round(float(np.mean(fold_r2s)),   3)
            cv_mae  = round(float(np.mean(fold_maes)),  3)
            if not silent:
                print(f"INFO: CV summary → "
                      f"RMSE={cv_rmse}±{np.std(fold_rmses):.3f}  "
                      f"R²={cv_r2}±{np.std(fold_r2s):.3f}  "
                      f"MAE={cv_mae}±{np.std(fold_maes):.3f}")

            # Final model
            if not silent:
                print(f"INFO: Fitting final model on all {len(X_all)} samples…")

            model_lgb = lgb.LGBMRegressor(
                n_estimators=500, max_depth=5,
                learning_rate=0.03, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8,
                min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
                verbosity=-1, random_state=42,
            )
            model_lgb.fit(X_all, y_all)

            # In-sample obs-space R² for sanity check
            train_ratio_preds = model_lgb.predict(X_all)
            if target_mode == "ratio" and raw_col in merged_train.columns:
                train_obs_preds = (merged_train[raw_col].values *
                                   np.clip(train_ratio_preds, 0.05, 20.0))
                train_obs_true  = merged_train["value"].values
            else:
                train_obs_preds = train_ratio_preds
                train_obs_true  = y_all

            metrics = {
                "CV_RMSE":    cv_rmse,
                "CV_R2":      cv_r2,
                "CV_MAE":     cv_mae,
                "RMSE":       cv_rmse,
                "R2":         cv_r2,
                "MAE":        cv_mae,
                "Train_R2":   round(r2_score(train_obs_true, train_obs_preds), 3),
                "n_folds":    n_splits,
                "n_train":    len(merged_train),
                "trained_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "target":     target_mode,
                "source":     "trained",
            }
            if not silent:
                print(f"INFO: Final Train R²={metrics['Train_R2']}  "
                      f"CV R²={cv_r2}  CV RMSE={cv_rmse}  target={target_mode}")

            _save_model(model_lgb, sel_feats, loc_clean, spec,
                        s3_manager, silent=silent, target_mode=target_mode, metrics=metrics)

        # Predict
        if not silent:
            print(f"INFO: Predicting over {len(geos_data)} GEOS-CF rows…")

        # Apply the same feature engineering
        geos_data_feat = _add_atmospheric_features(geos_data, spec=spec)
        all_geos = funcs.clean_feature_names(geos_data_feat.copy())

        # Align feature names with what was used at training time
        def _clean_feat(name: str) -> str:
            tmp = pd.DataFrame(columns=[name])
            return funcs.clean_feature_names(tmp).columns[0]

        sel_feats_clean = [_clean_feat(f) for f in sel_feats]
        sel_feats_avail = [f for f in sel_feats_clean if f in all_geos.columns]

        if not silent:
            print(f"INFO: {len(sel_feats_avail)}/{len(sel_feats_clean)} "
                  f"features available for prediction")

        if not sel_feats_avail:
            print(f"ERROR: No model features found in GEOS-CF data")
            return None, None, None

        all_geos["localised"] = np.nan

        # Diagnostics
        null_counts = all_geos[sel_feats_avail].isnull().sum()
        null_feats  = null_counts[null_counts > 0]
        if not silent:
            print(f"INFO: NaN counts per feature (non-zero only):\n{null_feats.to_string() if not null_feats.empty else '  none'}")
            print(f"INFO: Rows with ALL features present: "
                  f"{all_geos[sel_feats_avail].notnull().all(axis=1).sum()} / {len(all_geos)}")

        # Drop features that are entirely NaN 
        all_nan_feats = [f for f in sel_feats_avail
                         if all_geos[f].isnull().all()]
        if all_nan_feats:
            print(f"WARNING: Dropping {len(all_nan_feats)} fully-NaN features: {all_nan_feats}")
            sel_feats_avail = [f for f in sel_feats_avail if f not in all_nan_feats]

        # Forward-fill then backward-fill sparse features
        # maximise the number of predictable rows, then fall back to column
        # median for any remaining NaN.
        pred_df = all_geos[sel_feats_avail].copy()
        pred_df = pred_df.ffill().bfill()
        col_medians = pred_df.median()
        pred_df = pred_df.fillna(col_medians)

        valid_mask = pred_df.notnull().all(axis=1)
        if not silent:
            print(f"INFO: Valid rows after fill: {valid_mask.sum()} / {len(all_geos)}")

        if valid_mask.sum() > 0:
            X_pred = pred_df.loc[valid_mask, sel_feats_avail].copy()
            model_features = list(model_lgb.feature_name_)
            for feat in model_features:
                if feat not in X_pred.columns:
                    X_pred[feat] = 0.0
            X_pred = X_pred[model_features]
            raw_preds = model_lgb.predict(X_pred)

            # If model was trained on bias ratio, convert back to concentrations
            target_mode = metrics.get("target", "absolute")
            raw_col_clean = _clean_feat(spec)
            if target_mode == "ratio" and raw_col_clean in all_geos.columns:
                raw_conc = all_geos.loc[valid_mask, raw_col_clean].values
                preds = raw_conc * np.clip(raw_preds, 0.05, 20.0)
                if not silent:
                    print(f"INFO: ratio mode — median correction factor: "
                          f"{np.nanmedian(np.clip(raw_preds, 0.05, 20.0)):.3f}")
            else:
                preds = raw_preds

            all_geos.loc[valid_mask, "localised"] = preds
            if not silent:
                print(f"INFO: {len(preds)} predictions "
                      f"({np.nanmin(preds):.2f} – {np.nanmax(preds):.2f})")
        else:
            print(f"ERROR: No valid rows for prediction")
            return None, None, None

        # Interpolate
        n_before = all_geos["localised"].isna().sum()
        if n_before:
            all_geos["localised"] = all_geos["localised"].interpolate(
                method="linear", limit=6
            )
            if not silent:
                filled = n_before - all_geos["localised"].isna().sum()
                if filled:
                    print(f"INFO: Interpolated {filled} missing localised values")

        # Merge
        result = all_geos.merge(obs_data[["time", "value"]], on="time", how="left")
        if not silent:
            n_obs = result["value"].notna().sum()
            print(f"INFO: Final result: {len(result)} rows, "
                  f"{n_obs} with observations, "
                  f"{len(result) - n_obs} forecast-only")

        # Overall AQI
        result = funcs.calculate_overall_aqi(result)

        return result, metrics, model_lgb

    except Exception as exc:
        print(f"ERROR: Forecast generation failed for {loc}: {exc}")
        import traceback
        traceback.print_exc()
        return None, None, None




def plot_forecast(result: pd.DataFrame, spec: str = "no2",
                  metrics: dict = None, loc: str = "",
                  obs_col: str = "value", raw_col: str = None,
                  localised_col: str = "localised"):
    """
    Plot raw GEOS-CF, localised forecast, and observations on a shared time axis.

    Parameters
    ----------
    result : pd.DataFrame
        Output of ``get_localised_forecast``.  Must contain a ``time`` column.
    spec : str
        Species name used as raw GEOS-CF column (e.g. ``'no2'``).
    metrics : dict, optional
        Metrics dict returned by ``get_localised_forecast``.
    loc : str
        Site name for the plot title.
    obs_col : str
        Column with in-situ observations (default: ``'value'``).
    raw_col : str, optional
        Raw GEOS-CF column name. Defaults to ``spec``.
    localised_col : str
        Bias-corrected column name (default: ``'localised'``).
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("WARNING: matplotlib not available — skipping plot")
        return

    if raw_col is None:
        raw_col = spec

    df = result.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").set_index("time")

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    if raw_col in df.columns:
        ax.plot(df.index, df[raw_col], color="tab:orange", lw=0.8,
                alpha=0.7, label=f"GEOS-CF raw ({raw_col})")
    if localised_col in df.columns:
        ax.plot(df.index, df[localised_col], color="tab:green", lw=1.0,
                label="Localised (bias-corrected)")
    if obs_col in df.columns:
        obs_mask = df[obs_col].notna()
        ax.scatter(df.index[obs_mask], df.loc[obs_mask, obs_col],
                   color="tab:blue", s=4, alpha=0.8, label="Observations", zorder=3)

    title = f"Bias-corrected {spec.upper()} forecast — {loc}" if loc else \
            f"Bias-corrected {spec.upper()} forecast"
    if metrics:
        r2  = metrics.get("R2")
        rmse = metrics.get("RMSE")
        src  = metrics.get("source", "")
        if r2 is not None:
            title += f"\nR²={r2}  RMSE={rmse}  source={src}"
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Concentration (ppb)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Residuals
    ax2 = axes[1]
    if obs_col in df.columns and localised_col in df.columns:
        both = df[[obs_col, localised_col]].dropna()
        residuals = both[obs_col] - both[localised_col]
        ax2.bar(both.index, residuals, width=0.04, color="tab:purple",
                alpha=0.5, label="Obs − Localised")
        ax2.axhline(0, color="k", lw=0.8, ls="--")
        ax2.set_ylabel("Residual (ppb)")
        ax2.legend(loc="upper left", fontsize=9)
        ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()
    return fig


