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
import json
import gc
from datetime import datetime, timedelta
from typing import Optional
import datetime as dt
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, train_test_split, KFold
from sklearn.impute import SimpleImputer
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
from MLpred.pandora import read_pandora, _cache_path, _is_cache_fresh, DEFAULT_CACHE_HOURS, OBS_CACHE_DIR
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


# ---------------------------------------------------------------------------
# Global model
# ---------------------------------------------------------------------------

def train_global_model(
    locations: list,
    spec: str = "no2",
    st=None,
    ed=None,
    unit: str = "ppb",
    rmv_out: bool = True,
    obs_cache_hours: int = DEFAULT_CACHE_HOURS,
    force_obs_refresh: bool = False,
    model_max_age_days: int = 30,
    silent: bool = False,
    s3_manager: S3Manager = None,
) -> tuple:
    """
    Train a single **global** LightGBM model pooled across all supplied locations.

    The trained model can later be compared with per-site models to decide
    which generalises better for any given location.

    Parameters
    ----------
    locations : list of dict
        Each dict must contain at minimum:
          ``loc``      – human-readable site name
          ``lat``      – latitude
          ``lon``      – longitude
          ``obs_src``  – observation source (``'pandora'``, ``'local'``, …)
          ``obs_url``  – URL or path to the observation file
        Optional keys (all default to their ``get_localised_forecast`` equivalents):
          ``obs_val_col``, ``time_col``, ``date_fmt``, ``lat_col``, ``lon_col``
    spec : str
        Target species (``'no2'``, ``'o3'``, ``'pm25'``).
    st, ed : datetime
        Training date range (defaults: 2018-01-01 → today).
    unit : str
        Concentration unit (default: ``'ppb'``).
    rmv_out : bool
        Apply Isolation-Forest outlier removal per site before pooling.
    obs_cache_hours : int
        Pandora cache staleness threshold.
    force_obs_refresh : bool
        Delete and re-download observation caches.
    model_max_age_days : int
        If a recent global model exists locally, skip retraining.
    silent : bool
        Suppress verbose output.
    s3_manager : S3Manager, optional
        Pre-configured S3Manager.

    Returns
    -------
    model : LGBMRegressor or None
    metrics : dict or None
        Keys: ``global_CV_RMSE``, ``global_CV_R2``, ``global_CV_MAE``,
        ``per_site`` (list of per-site dicts), ``n_train``, ``n_sites``,
        ``trained_at``, ``target``.
    feature_names : list[str] or None
    """
    import json as _json

    if st is None:
        st = dt.datetime(2018, 1, 1)
    if ed is None:
        ed = dt.datetime.today()
    if s3_manager is None:
        s3_manager = S3Manager(bucket_name=S3_BUCKET)

    # ------------------------------------------------------------------
    # Paths for the global model
    # ------------------------------------------------------------------
    GLOBAL_TAG   = "global"
    model_path   = os.path.join(MODELS_LOCAL_DIR, f"lgbm_{GLOBAL_TAG}_{spec}_basic.joblib")
    feature_path = os.path.join(MODELS_LOCAL_DIR, f"lgbm_{GLOBAL_TAG}_{spec}_features_basic.pkl")
    metrics_path = os.path.join(MODELS_LOCAL_DIR, f"lgbm_{GLOBAL_TAG}_{spec}_metrics.json")

    # Check if a sufficiently fresh global model already exists
    if (
        os.path.exists(model_path) and
        os.path.exists(feature_path) and
        (dt.datetime.now().timestamp() - os.path.getmtime(model_path)) < model_max_age_days * 86400
    ):
        age_h = (dt.datetime.now().timestamp() - os.path.getmtime(model_path)) / 3600
        print(f"Global model is fresh ({age_h:.1f}h old). Loading from disk.")
        model = joblib.load(model_path)
        payload = pickle.load(open(feature_path, "rb"))
        sel_feats = payload["features"] if isinstance(payload, dict) else payload
        existing_metrics = {}
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path) as _f:
                    existing_metrics = _json.load(_f)
            except Exception:
                pass
        return model, existing_metrics, sel_feats

    # ------------------------------------------------------------------
    # Collect data from every location
    # ------------------------------------------------------------------
    all_frames   = []   # pooled training rows
    per_site_log = []   # per-site data-availability log

    print(f"\n{'='*60}")
    print(f"Training GLOBAL {spec.upper()} model  ({len(locations)} locations)")
    print(f"{'='*60}")

    for loc_cfg in locations:
        loc      = loc_cfg["loc"]
        lat      = float(loc_cfg["lat"])
        lon      = float(loc_cfg["lon"])
        obs_src  = loc_cfg.get("obs_src", "pandora")
        obs_url  = loc_cfg.get("obs_url", "")
        obs_val_col = loc_cfg.get("obs_val_col", "value")
        time_col    = loc_cfg.get("time_col",    "time")
        date_fmt    = loc_cfg.get("date_fmt",    "%Y-%m-%d %H:%M")
        lat_col     = loc_cfg.get("lat_col",     None)
        lon_col     = loc_cfg.get("lon_col",     None)

        print(f"\n  [{loc}]  lat={lat:.3f}  lon={lon:.3f}  src={obs_src}")

        try:
            # Observations
            obs_data = _load_observations(
                obs_src=obs_src, obs_url=obs_url, openaq_id=None,
                spec=spec, lat=lat, lon=lon, mod_src="s3", unit=unit,
                time_col=time_col, date_fmt=date_fmt, obs_val_col=obs_val_col,
                lat_col=lat_col, lon_col=lon_col, rmv_out=False,
                cache_hours=obs_cache_hours, force_refresh=force_obs_refresh,
                silent=silent,
            )
            if obs_data is None or obs_data.empty:
                print(f"    WARNING: No observations — skipping {loc}.")
                per_site_log.append({"loc": loc, "status": "no_obs", "n_rows": 0})
                continue

            # GEOS-CF V1 + V2
            geos_v1 = mlpred.read_geos_cf(lon=lon, lat=lat, start=st, end=ed,
                                           version=1, verbose=False)
            geos_v2 = mlpred.read_geos_cf(
                lon=lon, lat=lat,
                start=st,
                end=datetime.now() + timedelta(days=5),
                version=2, verbose=False,
            )
            frames_g = []
            for gv in (geos_v1, geos_v2):
                if gv is not None and not gv.empty:
                    gv["time"] = pd.to_datetime(gv["time"]).dt.floor("H")
                    frames_g.append(gv)
            if not frames_g:
                print(f"    WARNING: No GEOS-CF data — skipping {loc}.")
                per_site_log.append({"loc": loc, "status": "no_geos_cf", "n_rows": 0})
                continue

            geos_data = (
                pd.concat(frames_g, ignore_index=True)
                .drop_duplicates(subset=["time"], keep="first")
                .sort_values("time")
                .reset_index(drop=True)
            )

            # Merge obs + GEOS-CF
            merged = (
                geos_data.merge(obs_data[["time", "value"]], on="time", how="inner")
                         .dropna(subset=["value"])
            )
            if len(merged) < 20:
                print(f"    WARNING: Only {len(merged)} overlap rows — skipping {loc}.")
                per_site_log.append({"loc": loc, "status": "too_few_rows", "n_rows": len(merged)})
                continue

            # Feature engineering
            merged = _add_atmospheric_features(merged, spec=spec)

            # Bias ratio target (same logic as per-site training)
            raw_col = spec
            if raw_col not in merged.columns:
                hits = [c for c in merged.columns if spec.lower() in c.lower() and c != "value"]
                raw_col = hits[0] if hits else None

            if raw_col:
                raw_vals = merged[raw_col].replace(0, np.nan)
                merged["bias_ratio"] = (merged["value"] / raw_vals).clip(0.05, 20.0)
                merged = merged.dropna(subset=["bias_ratio"])
                target_mode = "ratio"
            else:
                target_mode = "absolute"

            # Outlier removal per site
            if rmv_out and len(merged) >= 50:
                yvar = "bias_ratio" if target_mode == "ratio" else "value"
                conc = merged[yvar].values.reshape(-1, 1)
                clf  = IForest(contamination=0.02)
                clf.fit(conc)
                mask = clf.predict(conc) != 1
                removed = (~mask).sum()
                merged = merged[mask]
                if not silent:
                    print(f"    Outlier removal: {removed} removed, {mask.sum()} remaining.")

            # Tag with site identity (enables per-site holdout CV later)
            merged["_site"] = loc
            all_frames.append(merged)
            per_site_log.append({"loc": loc, "status": "ok", "n_rows": len(merged)})
            print(f"    OK — {len(merged)} training rows.")

        except Exception as exc:
            import traceback as _tb
            print(f"    ERROR loading {loc}: {exc}")
            if not silent:
                _tb.print_exc()
            per_site_log.append({"loc": loc, "status": f"error: {exc}", "n_rows": 0})
            continue

    if not all_frames:
        print("ERROR: No usable data collected for global model training.")
        return None, None, None

    pool = (
        pd.concat(all_frames, ignore_index=True)
        .sort_values("time")
        .reset_index(drop=True)
    )
    n_sites_ok = sum(1 for s in per_site_log if s["status"] == "ok")
    print(f"\nPooled dataset: {len(pool):,} rows from {n_sites_ok} sites.")

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------
    skip      = {"time", "location", "lat", "lon", "value", "bias_ratio", "_site"}
    num_cols  = pool.select_dtypes(include=[np.number]).columns.tolist()
    sel_feats = [c for c in num_cols if c not in skip]

    target_mode = "ratio" if "bias_ratio" in pool.columns else "absolute"
    yvar        = "bias_ratio" if target_mode == "ratio" else "value"

    X_raw, y_raw = funcs.clean_data(pool[sel_feats], pool[yvar])
    X_all        = funcs.clean_feature_names(X_raw)
    y_all        = y_raw.values
    sel_feats_clean = list(X_all.columns)

    def _clean_feat(name: str) -> str:
        tmp = pd.DataFrame(columns=[name])
        return funcs.clean_feature_names(tmp).columns[0]

    raw_col_clean = _clean_feat(spec)

    # ------------------------------------------------------------------
    # Leave-one-site-out cross-validation
    # ------------------------------------------------------------------
    sites        = pool["_site"].values
    unique_sites = pool["_site"].unique().tolist()
    print(f"\nLeave-one-site-out CV over {len(unique_sites)} sites …")

    cv_rmses, cv_r2s, cv_maes = [], [], []
    site_metrics = []

    for hold_site in unique_sites:
        tr_mask  = sites != hold_site
        val_mask = sites == hold_site

        if val_mask.sum() < 10:
            continue

        tx, ty = X_all[tr_mask], y_all[tr_mask]
        vx, vy = X_all[val_mask], y_all[val_mask]

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

        # Convert back to observation space
        if target_mode == "ratio" and raw_col_clean in X_all.columns:
            raw_v    = X_all.loc[val_mask, raw_col_clean].values if raw_col_clean in X_all.columns \
                       else pool.loc[val_mask, spec].values if spec in pool.columns else None
            # fall back to the pool's raw column
            if raw_v is None:
                raw_v = pool.loc[val_mask, spec].values if spec in pool.columns else np.ones(len(pv))
            vy_obs = pool.loc[val_mask, "value"].values
            pv_obs = raw_v * np.clip(pv, 0.05, 20.0)
        else:
            vy_obs, pv_obs = vy, pv

        s_rmse = float(np.sqrt(mean_squared_error(vy_obs, pv_obs)))
        s_r2   = float(r2_score(vy_obs, pv_obs))
        s_mae  = float(mean_absolute_error(vy_obs, pv_obs))

        cv_rmses.append(s_rmse)
        cv_r2s.append(s_r2)
        cv_maes.append(s_mae)
        site_metrics.append({
            "site": hold_site,
            "n_val": int(val_mask.sum()),
            "RMSE": round(s_rmse, 3),
            "R2":   round(s_r2,   3),
            "MAE":  round(s_mae,  3),
        })
        if not silent:
            print(f"  Hold-out [{hold_site}]: "
                  f"RMSE={s_rmse:.3f}  R2={s_r2:.3f}  MAE={s_mae:.3f}  (n={val_mask.sum()})")

    global_cv_rmse = round(float(np.mean(cv_rmses)), 3) if cv_rmses else None
    global_cv_r2   = round(float(np.mean(cv_r2s)),   3) if cv_r2s   else None
    global_cv_mae  = round(float(np.mean(cv_maes)),  3) if cv_maes  else None

    print(f"\nGlobal LOSO-CV → "
          f"RMSE={global_cv_rmse}  R2={global_cv_r2}  MAE={global_cv_mae}")

    # ------------------------------------------------------------------
    # Final model trained on ALL pooled data
    # ------------------------------------------------------------------
    print(f"\nFitting final global model on {len(X_all):,} samples …")
    model_lgb = lgb.LGBMRegressor(
        n_estimators=500, max_depth=5,
        learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
        verbosity=-1, random_state=42,
    )
    model_lgb.fit(X_all, y_all)

    # In-sample sanity check
    train_preds = model_lgb.predict(X_all)
    if target_mode == "ratio" and spec in pool.columns:
        train_obs_preds = pool[spec].values * np.clip(train_preds, 0.05, 20.0)
        train_obs_true  = pool["value"].values
    else:
        train_obs_preds = train_preds
        train_obs_true  = y_all

    train_r2 = round(float(r2_score(train_obs_true, train_obs_preds)), 3)
    print(f"Final global model — in-sample R²={train_r2}  (LOSO CV R²={global_cv_r2})")

    # ------------------------------------------------------------------
    # Metrics bundle
    # ------------------------------------------------------------------
    metrics = {
        # Aggregate LOSO-CV
        "global_CV_RMSE": global_cv_rmse,
        "global_CV_R2":   global_cv_r2,
        "global_CV_MAE":  global_cv_mae,
        # Convenience aliases used by plot_forecast / generate.py
        "RMSE":    global_cv_rmse,
        "R2":      global_cv_r2,
        "MAE":     global_cv_mae,
        "Train_R2": train_r2,
        # Per-site breakdown
        "per_site":     site_metrics,
        "site_summary": per_site_log,
        # Book-keeping
        "n_train":    int(len(pool)),
        "n_sites":    n_sites_ok,
        "target":     target_mode,
        "spec":       spec,
        "source":     "global",
        "trained_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    os.makedirs(MODELS_LOCAL_DIR, exist_ok=True)
    try:
        joblib.dump(model_lgb, model_path)
        payload = {"features": sel_feats_clean, "target": target_mode}
        pickle.dump(payload, open(feature_path, "wb"))
        with open(metrics_path, "w") as _f:
            _json.dump(metrics, _f, indent=2)
        print(f"Global model saved → {model_path}")
    except Exception as exc:
        print(f"WARNING: Could not save global model locally: {exc}")

    try:
        model_s3_key   = f"{S3_MODELS_PREFIX}/lgbm_{GLOBAL_TAG}_{spec}_basic.joblib"
        feature_s3_key = f"{S3_MODELS_PREFIX}/lgbm_{GLOBAL_TAG}_{spec}_features_basic.pkl"
        metrics_s3_key = f"{S3_MODELS_PREFIX}/lgbm_{GLOBAL_TAG}_{spec}_metrics.json"
        s3_manager.upload_file(model_path,   model_s3_key)
        s3_manager.upload_file(feature_path, feature_s3_key)
        s3_manager.upload_file(metrics_path, metrics_s3_key)
        print(f"Global model uploaded to S3 → {model_s3_key}")
    except Exception as exc:
        print(f"WARNING: Could not upload global model to S3: {exc}")

    return model_lgb, metrics, sel_feats_clean


def train_global_model_from_local(
    spec: str = "no2",
    obs_dir: Optional[str] = None,
    geos_cf_dir: Optional[str] = None,
    st=None,
    ed=None,
    rmv_out: bool = True,
    model_max_age_days: int = 30,
    min_overlap_rows: int = 20,
    silent: bool = False,
    s3_manager: S3Manager = None,
) -> tuple:
    """
    Train a global LightGBM model using **only locally cached** observation
    and GEOS-CF files — no remote downloads.

    File discovery
    --------------
    Observations  : ``<obs_dir>/<Site>_<spec>.csv``
    GEOS-CF       : ``<geos_cf_dir>/loc_<lat>_<lon>_v1.csv`` (V1 preferred, V2 fallback)

    The merged training dataset is saved as a Parquet file alongside the model
    (``MODELS/global_<spec>_pool.parquet``) for future analysis or re-use.

    Returns
    -------
    model : LGBMRegressor or None
    metrics : dict or None
    feature_names : list[str] or None
    """
    import json as _json
    import glob

    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO     = os.path.dirname(_THIS_DIR)

    if obs_dir is None:
        obs_dir = OBS_CACHE_DIR
    if geos_cf_dir is None:
        geos_cf_dir = os.path.join(_REPO, "GEOS_CF")
    if s3_manager is None:
        s3_manager = S3Manager(bucket_name=S3_BUCKET)

    GLOBAL_TAG   = "global"
    model_path   = os.path.join(MODELS_LOCAL_DIR, f"lgbm_{GLOBAL_TAG}_{spec}_basic.joblib")
    feature_path = os.path.join(MODELS_LOCAL_DIR, f"lgbm_{GLOBAL_TAG}_{spec}_features_basic.pkl")
    metrics_path = os.path.join(MODELS_LOCAL_DIR, f"lgbm_{GLOBAL_TAG}_{spec}_metrics.json")
    pool_path    = os.path.join(MODELS_LOCAL_DIR, f"global_{spec}_pool.parquet")

    # ------------------------------------------------------------------
    # Early exit if fresh model exists
    # ------------------------------------------------------------------
    if (
        os.path.exists(model_path) and os.path.exists(feature_path) and
        (dt.datetime.now().timestamp() - os.path.getmtime(model_path)) < model_max_age_days * 86400
    ):
        age_h = (dt.datetime.now().timestamp() - os.path.getmtime(model_path)) / 3600
        print(f"Global {spec.upper()} model is fresh ({age_h:.1f}h old) — skipping retraining.")
        model = joblib.load(model_path)
        payload = pickle.load(open(feature_path, "rb"))
        sel_feats = payload["features"] if isinstance(payload, dict) else payload
        existing_metrics = {}
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path) as _f:
                    existing_metrics = _json.load(_f)
            except Exception:
                pass
        return model, existing_metrics, sel_feats

    # ------------------------------------------------------------------
    # Discover OBS files
    # ------------------------------------------------------------------
    pattern   = os.path.join(obs_dir, f"*_{spec.lower()}.csv")
    obs_files = sorted(glob.glob(pattern))
    if not obs_files:
        print(f"No OBS files found: {pattern}")
        return None, None, None

    print(f"Training global {spec.upper()} model — {len(obs_files)} OBS files found.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _coord_to_str(val: float) -> str:
        return f"{val:.6f}".replace(".", "_").replace("-", "m")

    def _find_geos_cf(lat: float, lon: float) -> pd.DataFrame:
        lat_s, lon_s = _coord_to_str(lat), _coord_to_str(lon)
        frames_g = []
        for ver in (1, 2):
            fpath = os.path.join(geos_cf_dir, f"loc_{lat_s}_{lon_s}_v{ver}.csv")
            if os.path.exists(fpath):
                try:
                    df = pd.read_csv(fpath, parse_dates=["time"])
                    df["time"] = pd.to_datetime(df["time"]).dt.floor("H")
                    frames_g.append(df)
                except Exception:
                    pass
        if not frames_g:
            return pd.DataFrame()
        return (
            pd.concat(frames_g, ignore_index=True)
            .drop_duplicates(subset=["time"], keep="first")
            .sort_values("time")
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Build pooled training set
    # ------------------------------------------------------------------
    all_frames   = []
    per_site_log = []
    skipped = ok = 0

    for obs_path in obs_files:
        site_name = os.path.basename(obs_path).replace(f"_{spec.lower()}.csv", "")
        try:
            obs = pd.read_csv(obs_path)
            obs["time"] = pd.to_datetime(obs["time"]).dt.floor("H")

            # Standardise value column
            if "value" not in obs.columns:
                for alt in (spec, spec.lower(), "concentration", "val", "obs"):
                    if alt in obs.columns:
                        obs = obs.rename(columns={alt: "value"})
                        break
            if "value" not in obs.columns:
                per_site_log.append({"loc": site_name, "status": "no_value_col", "n_rows": 0})
                skipped += 1
                continue

            obs = obs.dropna(subset=["value"])
            if obs.empty:
                per_site_log.append({"loc": site_name, "status": "all_nan", "n_rows": 0})
                skipped += 1
                continue

            if st is not None:
                obs = obs[obs["time"] >= pd.Timestamp(st)]
            if ed is not None:
                obs = obs[obs["time"] <= pd.Timestamp(ed)]

            # Resolve coordinates
            lat = lon = None
            for c in ("lat", "latitude", "Latitude"):
                if c in obs.columns:
                    v = obs[c].dropna()
                    if not v.empty:
                        lat = float(v.iloc[0])
                    break
            for c in ("lon", "longitude", "Longitude"):
                if c in obs.columns:
                    v = obs[c].dropna()
                    if not v.empty:
                        lon = float(v.iloc[0])
                    break

            if lat is None or lon is None:
                per_site_log.append({"loc": site_name, "status": "no_coords", "n_rows": 0})
                skipped += 1
                continue

            geos_data = _find_geos_cf(lat, lon)
            if geos_data.empty:
                per_site_log.append({"loc": site_name, "status": "no_geos_cf", "n_rows": 0})
                skipped += 1
                del obs
                continue

            # Merge obs onto GEOS-CF — keep only needed columns to limit RAM
            merged = (
                geos_data[
                    [c for c in geos_data.columns if c not in ("lat", "lon", "location")]
                ]
                .merge(obs[["time", "value"]], on="time", how="inner")
                .dropna(subset=["value"])
            )
            del geos_data, obs   # free raw data immediately

            if len(merged) < min_overlap_rows:
                per_site_log.append({"loc": site_name, "status": "too_few_rows", "n_rows": len(merged)})
                skipped += 1
                del merged
                continue

            merged = _add_atmospheric_features(merged, spec=spec)

            # Bias-ratio target
            raw_col = spec
            if raw_col not in merged.columns:
                hits = [c for c in merged.columns if spec.lower() in c.lower() and c != "value"]
                raw_col = hits[0] if hits else None

            if raw_col:
                raw_vals = merged[raw_col].replace(0, np.nan)
                merged["bias_ratio"] = (merged["value"] / raw_vals).clip(0.05, 20.0)
                merged = merged.dropna(subset=["bias_ratio"])
                target_mode = "ratio"
            else:
                target_mode = "absolute"

            if rmv_out and len(merged) >= 50:
                yvar_tmp = "bias_ratio" if target_mode == "ratio" else "value"
                conc = merged[yvar_tmp].values.reshape(-1, 1)
                clf  = IForest(contamination=0.02)
                clf.fit(conc)
                merged = merged[clf.predict(conc) != 1]

            merged["_site"] = site_name

            # ── Keep only numeric + sentinel columns; cast to float32 ──────
            # This is the main memory saving: drop string/object cols and
            # halve float precision before appending to all_frames.
            keep_cols = (
                ["time", "_site", "value", "bias_ratio"]
                + [c for c in merged.select_dtypes(include=[np.number]).columns
                   if c not in ("value", "bias_ratio")]
            )
            keep_cols = [c for c in keep_cols if c in merged.columns]
            merged = merged[keep_cols].copy()
            float_cols = merged.select_dtypes(include=["float64"]).columns
            merged[float_cols] = merged[float_cols].astype("float32")

            all_frames.append(merged)
            per_site_log.append({"loc": site_name, "status": "ok", "n_rows": len(merged)})
            ok += 1
            del merged

        except Exception as exc:
            per_site_log.append({"loc": site_name, "status": f"error: {exc}", "n_rows": 0})
            skipped += 1
            continue

    if not all_frames:
        print("No usable data collected — global model not trained.")
        return None, None, None

    import gc
    pool = (
        pd.concat(all_frames, ignore_index=True)
        .sort_values("time")
        .reset_index(drop=True)
    )
    del all_frames
    gc.collect()
    print(f"Pooled: {len(pool):,} rows from {ok} sites ({skipped} skipped).")

    # ------------------------------------------------------------------
    # Save pooled dataset for future reuse
    # ------------------------------------------------------------------
    os.makedirs(MODELS_LOCAL_DIR, exist_ok=True)
    try:
        pool.to_parquet(pool_path, index=False)
        if not silent:
            print(f"Pooled dataset saved → {pool_path}")
    except Exception as exc:
        if not silent:
            print(f"WARNING: Could not save pooled dataset: {exc}")

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------
    skip      = {"time", "location", "lat", "lon", "value", "bias_ratio", "_site"}
    num_cols  = pool.select_dtypes(include=[np.number]).columns.tolist()
    sel_feats = [c for c in num_cols if c not in skip]

    target_mode = "ratio" if "bias_ratio" in pool.columns else "absolute"
    yvar        = "bias_ratio" if target_mode == "ratio" else "value"

    X_raw, y_raw = funcs.clean_data(pool[sel_feats], pool[yvar])
    X_all        = funcs.clean_feature_names(X_raw)
    y_all        = y_raw.values
    sel_feats_clean = list(X_all.columns)

    def _clean_feat(name: str) -> str:
        tmp = pd.DataFrame(columns=[name])
        return funcs.clean_feature_names(tmp).columns[0]

    # ------------------------------------------------------------------
    # Leave-one-site-out CV
    # ------------------------------------------------------------------
    sites        = pool["_site"].values
    unique_sites = pool["_site"].unique().tolist()
    print(f"LOSO-CV over {len(unique_sites)} sites …")

    cv_rmses, cv_r2s, cv_maes = [], [], []
    site_metrics = []

    for hold_site in unique_sites:
        tr_mask  = sites != hold_site
        val_mask = sites == hold_site
        if val_mask.sum() < 10:
            continue

        _m = lgb.LGBMRegressor(
            n_estimators=500, max_depth=5, learning_rate=0.03, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            reg_alpha=0.1, reg_lambda=0.1, verbosity=-1, random_state=42,
        )
        _m.fit(
            X_all[tr_mask], y_all[tr_mask],
            eval_set=[(X_all[val_mask], y_all[val_mask])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        pv = _m.predict(X_all[val_mask])

        if target_mode == "ratio" and spec in pool.columns:
            raw_v  = pool.loc[val_mask, spec].values
            vy_obs = pool.loc[val_mask, "value"].values
            pv_obs = raw_v * np.clip(pv, 0.05, 20.0)
        else:
            vy_obs = y_all[val_mask]
            pv_obs = pv

        s_rmse = float(np.sqrt(mean_squared_error(vy_obs, pv_obs)))
        s_r2   = float(r2_score(vy_obs, pv_obs))
        s_mae  = float(mean_absolute_error(vy_obs, pv_obs))
        cv_rmses.append(s_rmse); cv_r2s.append(s_r2); cv_maes.append(s_mae)
        site_metrics.append({
            "site": hold_site, "n_val": int(val_mask.sum()),
            "RMSE": round(s_rmse, 3), "R2": round(s_r2, 3), "MAE": round(s_mae, 3),
        })

    global_cv_rmse = round(float(np.mean(cv_rmses)), 3) if cv_rmses else None
    global_cv_r2   = round(float(np.mean(cv_r2s)),   3) if cv_r2s   else None
    global_cv_mae  = round(float(np.mean(cv_maes)),  3) if cv_maes  else None
    print(f"LOSO-CV → RMSE={global_cv_rmse}  R2={global_cv_r2}  MAE={global_cv_mae}")

    # ------------------------------------------------------------------
    # Final model on all pooled data
    # ------------------------------------------------------------------
    model_lgb = lgb.LGBMRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=0.1, verbosity=-1, random_state=42,
    )
    model_lgb.fit(X_all, y_all)

    train_preds = model_lgb.predict(X_all)
    if target_mode == "ratio" and spec in pool.columns:
        train_obs_preds = pool[spec].values * np.clip(train_preds, 0.05, 20.0)
        train_obs_true  = pool["value"].values
    else:
        train_obs_preds, train_obs_true = train_preds, y_all

    train_r2 = round(float(r2_score(train_obs_true, train_obs_preds)), 3)
    print(f"Final model — train R²={train_r2}  CV R²={global_cv_r2}")

    # ------------------------------------------------------------------
    # Metrics bundle
    # ------------------------------------------------------------------
    metrics = {
        "global_CV_RMSE": global_cv_rmse,
        "global_CV_R2":   global_cv_r2,
        "global_CV_MAE":  global_cv_mae,
        "RMSE":      global_cv_rmse,
        "R2":        global_cv_r2,
        "MAE":       global_cv_mae,
        "Train_R2":  train_r2,
        "per_site":  site_metrics,
        "site_summary": per_site_log,
        "n_train":   int(len(pool)),
        "n_sites":   ok,
        "target":    target_mode,
        "spec":      spec,
        "source":    "global_local",
        "obs_dir":   obs_dir,
        "geos_cf_dir": geos_cf_dir,
        "pool_path": pool_path,
        "trained_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    # ------------------------------------------------------------------
    # Persist locally + upload to S3
    # ------------------------------------------------------------------
    try:
        joblib.dump(model_lgb, model_path)
        pickle.dump({"features": sel_feats_clean, "target": target_mode}, open(feature_path, "wb"))
        with open(metrics_path, "w") as _f:
            _json.dump(metrics, _f, indent=2)
        print(f"Model saved → {model_path}")
    except Exception as exc:
        print(f"WARNING: Could not save model: {exc}")

    try:
        s3_manager.upload_file(model_path,   f"{S3_MODELS_PREFIX}/lgbm_{GLOBAL_TAG}_{spec}_basic.joblib")
        s3_manager.upload_file(feature_path, f"{S3_MODELS_PREFIX}/lgbm_{GLOBAL_TAG}_{spec}_features_basic.pkl")
        s3_manager.upload_file(metrics_path, f"{S3_MODELS_PREFIX}/lgbm_{GLOBAL_TAG}_{spec}_metrics.json")
        if not silent:
            print(f"Model uploaded to S3.")
    except Exception as exc:
        if not silent:
            print(f"WARNING: S3 upload failed: {exc}")

    return model_lgb, metrics, sel_feats_clean


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
        # Observations
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

        # GEOS-CF V1
        if not silent:
            print(f"INFO: Loading GEOS-CF V1 (replay)…")
        geos_v1 = mlpred.read_geos_cf(lon=lon, lat=lat, start=st, end=ed,
                                       version=1, verbose=not silent)

        # GEOS-CF V2 
        if not silent:
            print(f"INFO: Loading GEOS-CF V2 (analysis + forecast)…")
        geos_v2 = mlpred.read_geos_cf(
            lon=lon, lat=lat,
            start=st,
            end=datetime.now() + timedelta(days=5),
            version=2, verbose=not silent,
        )

        # Combine V1 + V2 
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

        # Model
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

            if isinstance(payload, dict):
                sel_feats   = payload["features"]
                target_mode_loaded = payload.get("target", "absolute")
            else:
                sel_feats   = payload
                target_mode_loaded = "absolute"
            metrics = {"source": "local", "target": target_mode_loaded}
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


        if model_lgb is not None and metrics.get("RMSE") is None:
            try:
                merged_eval = (
                    geos_data.merge(obs_data[["time", "value"]], on="time", how="inner")
                             .dropna(subset=["value"])
                )
                if len(merged_eval) >= 10:
                    # Apply feature engineering as training
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

        # Train 
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


        geos_data_feat = _add_atmospheric_features(geos_data, spec=spec)
        all_geos = funcs.clean_feature_names(geos_data_feat.copy())

 
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


# ── GLOBAL MODEL TRAINING ──────────────────────────────────────────────────────
def train_global(
    data_path: str = "../MODELS/global_no2_dataset.parquet",
    model_output_path: str = "../MODELS/global_no2_lgbm_improved.joblib",
    target_variable: str = "value",
    max_rows: int = 500000,
    n_splits: int = 3,
    random_state: int = 42,
    verbose: bool = True
) -> dict:
    """
    Train memory-optimized global NO2 LightGBM model with lag features.
    
    Optimized for 16GB RAM systems. Creates lag features (6h, 12h, 24h),
    rolling statistics, cyclical temporal encoding, and k-fold cross-validation.
    
    Parameters
    ----------
    data_path : str
        Path to parquet file with global NO2 dataset
    model_output_path : str
        Where to save trained LightGBM model
    target_variable : str
        Column name for target variable (observations)
    max_rows : int
        Maximum rows to process (for memory efficiency)
    n_splits : int
        Number of folds for cross-validation
    random_state : int
        Random seed for reproducibility
    verbose : bool
        Print training progress
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'model': Trained LightGBM model
        - 'metadata': Training metadata (features, metrics, params)
        - 'model_path': Path where model was saved
        - 'metadata_path': Path where metadata was saved
    
    Examples
    --------
    >>> result = train_global(
    ...     data_path="global_no2_dataset.parquet",
    ...     model_output_path="my_model.joblib"
    ... )
    >>> print(f"Test R²: {result['metadata']['test_r2']:.4f}")
    >>> model = result['model']
    """
    if verbose:
        print("="*70)
        print("MEMORY-OPTIMIZED GLOBAL NO2 MODEL TRAINING")
        print("="*70)
    
    # ── CONFIG ──────────────────────────────────────────────────────────────────
    FEATURES = [
        "no2", "o3", "co", "hcho", "no", "noy", "so2",
        "pm25_rh35", "pm25_rh35_gcc", "pm10_rh35",
        "pm25bc_rh35", "pm25du_rh35", "pm25nh4_rh35", "pm25nit_rh35",
        "pm25oc_rh35", "pm25soa_rh35", "pm25ss_rh35", "pm25su_rh35",
        "aod550_bc", "aod550_dust", "aod550_oc", "aod550_sala", "aod550_salc",
        "aod550_sulfate", "aod550_psc", "aod550_sla", "aod550_sna", "aod550_ss",
        "cldtt", "lev", "ps", "rh", "t10m", "tprec", "zpbl",
    ]
    
    # ── LOAD DATA ───────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n✓ Loading dataset from {data_path}...")
    
    df = pd.read_parquet(data_path, dtype_backend='numpy_nullable')
    if verbose:
        print(f"✓ Dataset: {len(df):,} rows × {df.shape[1]} columns")
    
    # Optimize dtypes
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif col == '_site':
            df[col] = df[col].astype('category')
    
    gc.collect()
    if verbose:
        print(f"✓ Memory after optimization: {df.memory_usage(deep=True).sum() / (1024**3):.2f} GB")
    
    # Remove NaN targets
    df = df[df[target_variable].notna()].copy()
    if verbose:
        print(f"✓ After removing NaN targets: {len(df):,} rows")
    
    # ── TEMPORAL FEATURES ───────────────────────────────────────────────────────
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors='coerce')
        df["hour"] = df["time"].dt.hour.astype('int8')
        df["doy"] = df["time"].dt.dayofyear.astype('int16')
        df["weekday"] = df["time"].dt.weekday.astype('int8')
        df["month"] = df["time"].dt.month.astype('int8')
        
        # Cyclical encoding
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype('float32')
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype('float32')
        df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365).astype('float32')
        df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365).astype('float32')
        
        df = df.drop(columns=['time'])
        df = df[df["hour"].notna()].copy()
        if verbose:
            print(f"✓ After temporal features: {len(df):,} rows")
    
    gc.collect()
    
    # ── LAG FEATURES ────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n--- CREATING LAG FEATURES ---")
    
    # Sample if needed
    if len(df) > max_rows:
        if verbose:
            print(f"⚠️  Sampling {len(df):,} rows → {max_rows:,}")
        if "_site" in df.columns:
            df = df.groupby('_site', group_keys=False).apply(
                lambda x: x.sample(
                    min(len(x), max(1, int(max_rows * len(x) / len(df)))),
                    random_state=random_state
                )
            ).reset_index(drop=True)
        else:
            df = df.sample(n=max_rows, random_state=random_state).sort_index()
    
    # Create lags grouped by site
    lag_features = []
    if "_site" in df.columns:
        df = df.sort_values(["_site", "hour"]).reset_index(drop=True)
        
        for lag in [6, 12, 24]:
            col_name = f'{target_variable}_lag{lag}h'
            df[col_name] = df.groupby('_site')[target_variable].shift(lag).astype('float32')
            lag_features.append(col_name)
        
        for window in [6, 12]:
            col_name = f'{target_variable}_roll_mean_{window}h'
            df[col_name] = df.groupby('_site')[target_variable].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            ).astype('float32')
            lag_features.append(col_name)
        
        if verbose:
            print(f"✓ Created {len(lag_features)} lag/rolling features")
    
    gc.collect()
    
    # ── FEATURE FILTERING ───────────────────────────────────────────────────────
    temporal_features = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    available_features = [f for f in FEATURES + temporal_features + lag_features 
                          if f in df.columns]
    
    if verbose:
        print(f"\n✓ Total features: {len(available_features)}")
        print(f"  - Base: {len([f for f in FEATURES if f in df.columns])}")
        print(f"  - Temporal: {len(temporal_features)}")
        print(f"  - Lag/Rolling: {len(lag_features)}")
    
    # ── DATA CLEANING ───────────────────────────────────────────────────────────
    if verbose:
        print(f"\n--- DATA CLEANING ---")
    
    X_raw = df[available_features].copy()
    y = df[target_variable].copy()
    
    if verbose:
        print(f"Before imputation: {X_raw.isnull().sum().sum():,} NaN values")
    
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X_raw),
        columns=available_features,
        index=X_raw.index
    ).astype('float32')
    
    del X_raw
    gc.collect()
    
    # Outlier removal
    target_mean = y.mean()
    target_std = y.std()
    outlier_mask = (y >= target_mean - 5*target_std) & (y <= target_mean + 5*target_std)
    if verbose:
        print(f"Removing {(~outlier_mask).sum():,} outliers (>5σ)")
    
    X_clean = X_imputed[outlier_mask].copy()
    y_clean = y[outlier_mask].copy()
    
    del X_imputed, df, y
    gc.collect()
    
    if verbose:
        print(f"✓ Clean dataset: {len(X_clean):,} rows")
    
    # ── TRAIN/TEST SPLIT ────────────────────────────────────────────────────────
    if verbose:
        print(f"\n" + "="*70)
        print("TRAINING WITH K-FOLD CROSS-VALIDATION")
        print("="*70)
    
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_clean, y_clean,
        test_size=0.15,
        random_state=random_state,
        shuffle=True
    )
    
    del X_clean, y_clean
    gc.collect()
    
    if verbose:
        print(f"\nTrain: {len(X_tr):,} rows | Test: {len(X_te):,} rows")
    
    # ── MODEL CONFIG ────────────────────────────────────────────────────────────
    model_params = {
        'n_estimators': 300,
        'learning_rate': 0.02,
        'max_depth': 8,
        'num_leaves': 64,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 20,
        'reg_alpha': 0.2,
        'reg_lambda': 0.2,
        'feature_fraction': 0.7,
        'verbosity': -1,
        'random_state': random_state,
        'n_jobs': 2,
    }
    
    if verbose:
        print(f"\n--- MODEL PARAMS ---")
        for key, val in model_params.items():
            if key not in ['verbosity', 'random_state', 'n_jobs']:
                print(f"  {key}: {val}")
    
    # ── K-FOLD CROSS-VALIDATION ────────────────────────────────────────────────
    if verbose:
        print(f"\n--- K-FOLD CV ({n_splits} Folds) ---")
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_scores = {'r2': [], 'rmse': [], 'mae': []}
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_tr), 1):
        X_fold_tr, X_fold_val = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
        y_fold_tr, y_fold_val = y_tr.iloc[train_idx], y_tr.iloc[val_idx]
        
        model_fold = lgb.LGBMRegressor(**model_params)
        model_fold.fit(
            X_fold_tr, y_fold_tr,
            eval_set=[(X_fold_val, y_fold_val)],
            callbacks=[
                lgb.early_stopping(20, verbose=False),
                lgb.log_evaluation(0)
            ]
        )
        
        y_fold_pred = model_fold.predict(X_fold_val)
        fold_r2 = r2_score(y_fold_val, y_fold_pred)
        fold_rmse = np.sqrt(mean_squared_error(y_fold_val, y_fold_pred))
        fold_mae = mean_absolute_error(y_fold_val, y_fold_pred)
        
        cv_scores['r2'].append(fold_r2)
        cv_scores['rmse'].append(fold_rmse)
        cv_scores['mae'].append(fold_mae)
        
        if verbose:
            print(f"  Fold {fold}: R²={fold_r2:.4f} | RMSE={fold_rmse:.4f} | MAE={fold_mae:.4f}")
        
        del model_fold, X_fold_tr, X_fold_val, y_fold_tr, y_fold_val
        gc.collect()
    
    if verbose:
        print(f"\n--- CV SUMMARY ---")
        print(f"  R²:   {np.mean(cv_scores['r2']):.4f} ± {np.std(cv_scores['r2']):.4f}")
        print(f"  RMSE: {np.mean(cv_scores['rmse']):.4f} ± {np.std(cv_scores['rmse']):.4f} ppb")
        print(f"  MAE:  {np.mean(cv_scores['mae']):.4f} ± {np.std(cv_scores['mae']):.4f} ppb")
    
    # ── TRAIN FINAL MODEL ───────────────────────────────────────────────────────
    if verbose:
        print(f"\n--- FINAL MODEL ---")
        print(f"Training on {len(X_tr):,} samples...")
    
    model = lgb.LGBMRegressor(**model_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        callbacks=[
            lgb.early_stopping(40, verbose=False),
            lgb.log_evaluation(0 if not verbose else 50)
        ]
    )
    
    # ── EVALUATION ──────────────────────────────────────────────────────────────
    y_tr_pred = model.predict(X_tr)
    y_te_pred = model.predict(X_te)
    
    tr_r2 = r2_score(y_tr, y_tr_pred)
    tr_rmse = np.sqrt(mean_squared_error(y_tr, y_tr_pred))
    tr_mae = mean_absolute_error(y_tr, y_tr_pred)
    
    te_r2 = r2_score(y_te, y_te_pred)
    te_rmse = np.sqrt(mean_squared_error(y_te, y_te_pred))
    te_mae = mean_absolute_error(y_te, y_te_pred)
    
    if verbose:
        print(f"\n" + "="*70)
        print("FINAL EVALUATION")
        print("="*70)
        print(f"\n--- TRAIN METRICS ---")
        print(f"  R²:   {tr_r2:.4f}")
        print(f"  RMSE: {tr_rmse:.4f} ppb")
        print(f"  MAE:  {tr_mae:.4f} ppb")
        print(f"\n--- TEST METRICS ---")
        print(f"  R²:   {te_r2:.4f} ✓")
        print(f"  RMSE: {te_rmse:.4f} ppb ✓")
        print(f"  MAE:  {te_mae:.4f} ppb ✓")
        
        rmse_gap = abs(tr_rmse - te_rmse)
        r2_gap = abs(tr_r2 - te_r2)
        print(f"\n--- GENERALIZATION ---")
        print(f"  RMSE gap: {rmse_gap:.4f}")
        print(f"  R² gap:   {r2_gap:.4f}")
    
    # ── SAVE MODEL ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    if verbose:
        print(f"\n✓ Model saved → {model_output_path}")
    
    # Save metadata
    metadata = {
        'model_name': 'global_no2_lgbm_improved',
        'model_path': model_output_path,
        'target_variable': target_variable,
        'features': available_features,
        'lag_features': lag_features,
        'temporal_features': temporal_features,
        'n_features': len(available_features),
        'test_r2': float(te_r2),
        'test_rmse': float(te_rmse),
        'test_mae': float(te_mae),
        'cv_r2_mean': float(np.mean(cv_scores['r2'])),
        'cv_rmse_mean': float(np.mean(cv_scores['rmse'])),
        'cv_mae_mean': float(np.mean(cv_scores['mae'])),
        'train_samples': len(X_tr),
        'test_samples': len(X_te),
        'model_params': model_params,
    }
    
    metadata_path = model_output_path.replace('.joblib', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    if verbose:
        print(f"✓ Metadata saved → {metadata_path}")
    
    if verbose:
        print(f"\n" + "="*70)
        print("✓ TRAINING COMPLETE")
        print("="*70)
    
    return {
        'model': model,
        'metadata': metadata,
        'model_path': model_output_path,
        'metadata_path': metadata_path
    }


# ── ENHANCED LOCALISED FORECAST WITH GLOBAL FALLBACK ──────────────────────────
def get_localised_forecast_v2(
    loc: str,
    spec: str = "no2",
    lat: float = 0.0,
    lon: float = 0.0,
    mod_src: str = "s3",
    obs_src: str = "pandora",
    OBS_URL: str = None,
    st=None,
    ed=None,
    resamp: str = "1h",
    unit: str = "ppb",
    interpol: bool = True,
    rmv_out: bool = True,
    time_col: str = "time",
    date_fmt: str = "%Y-%m-%d %H:%M",
    obs_val_col: str = "value",
    lat_col=None,
    lon_col=None,
    silent: bool = False,
    force_retrain: bool = False,
    force_obs_refresh: bool = False,
    model_max_age_days: int = 7,
    obs_cache_hours: int = DEFAULT_CACHE_HOURS,
    r2_threshold: float = 0.50,
    global_model_path: str = None,
    global_metadata_path: str = None,
    s3_manager: S3Manager = None,
    **kwargs,
):
    """
    Enhanced localised forecast with global model fallback.
    
    Workflow
    --------
    1. **Load observations** – via ``read_pandora`` or ``mlpred.ObsSite``
    2. **Load GEOS-CF data** – V1 (replay) + V2 (analysis/forecast)
    3. **Train local model** – with lag features, cyclical encoding, k-fold CV
    4. **Evaluate local model** – compute R² on validation data
    5. **Fallback decision** – if local R² < threshold, use global model instead
    6. **Generate predictions** – using selected model
    7. **Interpolate & merge** – with observations for comparison
    
    Parameters
    ----------
    loc : str
        Human-readable location name
    spec : str
        Target species ('no2', 'o3', 'pm25')
    lat, lon : float
        Site coordinates
    mod_src : str
        GEOS-CF data source ('s3' or 'local')
    obs_src : str
        Observation source ('pandora', 'openaq', 'local')
    OBS_URL : str
        URL or file path to observations
    r2_threshold : float
        If local model R² falls below this, use global model (default: 0.50)
    global_model_path : str
        Path to pre-trained global model (e.g., from train_global())
    global_metadata_path : str
        Path to global model metadata JSON
    
    Returns
    -------
    result : pd.DataFrame or None
        Forecast with localised + global predictions
    metrics : dict or None
        Model performance metrics and metadata
    model : LGBMRegressor or None
        The selected model (local or global)
    """
    import traceback
    
    if st is None:
        st = dt.datetime(2018, 1, 1)
    if ed is None:
        ed = dt.datetime.today()
    if s3_manager is None:
        s3_manager = S3Manager(bucket_name=S3_BUCKET)
    
    if not silent:
        print("="*70)
        print(f"ENHANCED LOCALISED FORECAST: {loc} ({spec.upper()})")
        print("="*70)
    
    try:
        # ────────────────────────────────────────────────────────────────────
        # STEP 1: LOAD OBSERVATIONS
        # ────────────────────────────────────────────────────────────────────
        if not silent:
            print(f"\n--- LOADING OBSERVATIONS ---")
            print(f"Source: {obs_src} → {OBS_URL}")
        
        obs_data = _load_observations(
            obs_src=obs_src, obs_url=OBS_URL, openaq_id=None,
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
            print(f"✓ Loaded {len(obs_data)} observations")
            print(f"  Date range: {obs_data['time'].min()} → {obs_data['time'].max()}")
        
        # ────────────────────────────────────────────────────────────────────
        # STEP 2: LOAD GEOS-CF DATA
        # ────────────────────────────────────────────────────────────────────
        if not silent:
            print(f"\n--- LOADING GEOS-CF DATA ---")
        
        # V1 (replay)
        geos_v1 = mlpred.read_geos_cf(
            lon=lon, lat=lat, start=st, end=ed, version=1, verbose=not silent
        )
        
        # V2 (analysis + forecast)
        geos_v2 = mlpred.read_geos_cf(
            lon=lon, lat=lat, start=st, end=datetime.now() + timedelta(days=5),
            version=2, verbose=not silent
        )
        
        # Combine V1 + V2
        frames = []
        if geos_v1 is not None and not geos_v1.empty:
            geos_v1["time"] = pd.to_datetime(geos_v1["time"]).dt.floor("H")
            frames.append(geos_v1)
            if not silent:
                print(f"✓ V1: {len(geos_v1)} rows "
                      f"({geos_v1['time'].min()} → {geos_v1['time'].max()})")
        
        if geos_v2 is not None and not geos_v2.empty:
            geos_v2["time"] = pd.to_datetime(geos_v2["time"]).dt.floor("H")
            frames.append(geos_v2)
            if not silent:
                print(f"✓ V2: {len(geos_v2)} rows "
                      f"({geos_v2['time'].min()} → {geos_v2['time'].max()})")
        
        if not frames:
            print(f"ERROR: No GEOS-CF data available for {loc}")
            return None, None, None
        
        geos_data = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=["time"], keep="first")
            .sort_values("time")
            .reset_index(drop=True)
        )
        if not silent:
            print(f"✓ Combined: {len(geos_data)} rows")
        
        # ────────────────────────────────────────────────────────────────────
        # STEP 3: MERGE FOR TRAINING
        # ────────────────────────────────────────────────────────────────────
        if not silent:
            print(f"\n--- MERGING OBSERVATIONS & GEOS-CF ---")
        
        merged_train = (
            geos_data.merge(obs_data[["time", "value"]], on="time", how="inner")
            .dropna(subset=["value"])
        )
        
        if len(merged_train) < 50:
            print(f"WARNING: Only {len(merged_train)} training samples (need ≥50)")
            print(f"  Using global model as fallback")
            merged_train = None
        else:
            if not silent:
                print(f"✓ Merged: {len(merged_train)} rows")
        
        # ────────────────────────────────────────────────────────────────────
        # STEP 4: ADD ENGINEERED FEATURES
        # ────────────────────────────────────────────────────────────────────
        if merged_train is not None:
            if not silent:
                print(f"\n--- ENGINEERING FEATURES ---")
            
            merged_train = _add_atmospheric_features(merged_train, spec=spec)
            
            if not silent:
                new_feat_count = len([c for c in merged_train.columns
                                      if c not in geos_data.columns and c != "value"])
                print(f"✓ Added {new_feat_count} engineered features")
                print(f"  - Cyclical time (hour_sin/cos, doy_sin/cos)")
                print(f"  - Lag features (lag1h, lag3h, lag6h, lag24h)")
                print(f"  - Rolling statistics (roll3h, roll6h)")
                print(f"  - Atmospheric proxies (pbl_proxy, t2m_delta, etc.)")
        
        # ────────────────────────────────────────────────────────────────────
        # STEP 5: TRAIN LOCAL MODEL
        # ────────────────────────────────────────────────────────────────────
        model_local = None
        local_r2 = None
        local_metrics = {}
        
        if merged_train is not None and len(merged_train) >= 50:
            if not silent:
                print(f"\n--- TRAINING LOCAL MODEL ---")
            
            # Feature selection
            skip = {"time", "location", "lat", "lon", "value", "bias_ratio"}
            num_cols = merged_train.select_dtypes(include=[np.number]).columns.tolist()
            sel_feats = [c for c in num_cols if c not in skip]
            
            # Create target (bias ratio if possible, else absolute)
            raw_col = spec
            if raw_col not in merged_train.columns:
                hits = [c for c in merged_train.columns
                        if spec.lower() in c.lower() and c != "value"]
                raw_col = hits[0] if hits else None
            
            if raw_col and raw_col in merged_train.columns:
                raw_vals = merged_train[raw_col].replace(0, np.nan)
                merged_train["bias_ratio"] = (merged_train["value"] / raw_vals).clip(0.05, 20.0)
                merged_train = merged_train.dropna(subset=["bias_ratio"])
                yvar = "bias_ratio"
                target_mode = "ratio"
                if not silent:
                    print(f"✓ Target: bias_ratio (obs / {raw_col})")
                    print(f"  Median ratio: {merged_train['bias_ratio'].median():.3f}")
            else:
                yvar = "value"
                target_mode = "absolute"
                if not silent:
                    print(f"✓ Target: absolute concentration")
            
            # Outlier removal
            if rmv_out and len(merged_train) >= 50:
                conc = merged_train[yvar].values.reshape(-1, 1)
                clf = IForest(contamination=0.02)
                clf.fit(conc)
                mask = clf.predict(conc) != 1
                if not silent:
                    print(f"✓ Outlier removal: {(~mask).sum()} removed")
                merged_train = merged_train[mask]
            
            # Clean features
            X_raw, y_raw = funcs.clean_data(merged_train[sel_feats], merged_train[yvar])
            X_all = funcs.clean_feature_names(X_raw)
            y_all = y_raw.values
            sel_feats_clean = list(X_all.columns)
            
            if not silent:
                print(f"✓ Features: {len(sel_feats_clean)}")
                print(f"  Training samples: {len(X_all)}")
            
            # K-fold CV
            n_splits = min(3, max(2, len(merged_train) // 300))
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_r2s = []
            
            if not silent:
                print(f"\n--- K-FOLD CROSS-VALIDATION ({n_splits} folds) ---")
            
            for fold, (tr_idx, val_idx) in enumerate(kfold.split(X_all), 1):
                X_tr_fold, X_val_fold = X_all.iloc[tr_idx], X_all.iloc[val_idx]
                y_tr_fold, y_val_fold = y_all[tr_idx], y_all[val_idx]
                
                m_fold = lgb.LGBMRegressor(
                    n_estimators=300, learning_rate=0.02, max_depth=8,
                    num_leaves=64, subsample=0.8, colsample_bytree=0.8,
                    min_child_samples=20, reg_alpha=0.2, reg_lambda=0.2,
                    verbosity=-1, random_state=42, n_jobs=2,
                )
                m_fold.fit(
                    X_tr_fold, y_tr_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    callbacks=[
                        lgb.early_stopping(20, verbose=False),
                        lgb.log_evaluation(0)
                    ]
                )
                
                y_fold_pred = m_fold.predict(X_val_fold)
                fold_r2 = r2_score(y_val_fold, y_fold_pred)
                cv_r2s.append(fold_r2)
                
                if not silent:
                    print(f"  Fold {fold}: R² = {fold_r2:.4f}")
                
                del m_fold, X_tr_fold, X_val_fold, y_tr_fold, y_val_fold
                gc.collect()
            
            local_r2 = float(np.mean(cv_r2s))
            if not silent:
                print(f"\n✓ Local CV R² = {local_r2:.4f} ± {np.std(cv_r2s):.4f}")
            
            # Train final local model
            if not silent:
                print(f"\n--- TRAINING FINAL LOCAL MODEL ---")
            
            model_local = lgb.LGBMRegressor(
                n_estimators=300, learning_rate=0.02, max_depth=8,
                num_leaves=64, subsample=0.8, colsample_bytree=0.8,
                min_child_samples=20, reg_alpha=0.2, reg_lambda=0.2,
                verbosity=-1, random_state=42, n_jobs=2,
            )
            model_local.fit(X_all, y_all)
            
            local_metrics = {
                'source': 'local',
                'cv_r2': float(local_r2),
                'n_train': len(merged_train),
                'n_features': len(sel_feats_clean),
                'target_mode': target_mode,
                'features': sel_feats_clean,
            }
            
            if not silent:
                print(f"✓ Final local model trained")
        
        # ────────────────────────────────────────────────────────────────────
        # STEP 6: MODEL SELECTION (LOCAL vs GLOBAL)
        # ────────────────────────────────────────────────────────────────────
        if not silent:
            print(f"\n--- MODEL SELECTION ---")
        
        use_global = False
        selected_model = None
        selected_metrics = {}
        
        if local_r2 is not None and local_r2 >= r2_threshold:
            if not silent:
                print(f"✓ LOCAL model selected (R² = {local_r2:.4f} ≥ {r2_threshold})")
            selected_model = model_local
            selected_metrics = local_metrics
        else:
            if not silent:
                if local_r2 is not None:
                    print(f"✗ Local R² = {local_r2:.4f} < {r2_threshold} threshold")
                else:
                    print(f"✗ Insufficient training data")
                print(f"  → Falling back to GLOBAL model")
            use_global = True
        
        # Load global model if needed
        global_features = None
        if use_global:
            if global_model_path is None:
                global_model_path = os.path.join(MODELS_LOCAL_DIR, 
                                                 f"global_no2_lgbm_improved.joblib")
            if global_metadata_path is None:
                global_metadata_path = os.path.join(MODELS_LOCAL_DIR,
                                                    f"global_no2_lgbm_improved_metadata.json")
            
            if os.path.exists(global_model_path) and os.path.exists(global_metadata_path):
                try:
                    selected_model = joblib.load(global_model_path)
                    with open(global_metadata_path, 'r') as f:
                        global_meta = json.load(f)
                    global_features = global_meta.get('features', [])
                    selected_metrics = {
                        'source': 'global',
                        'test_r2': global_meta.get('test_r2'),
                        'test_rmse': global_meta.get('test_rmse'),
                        'cv_r2': global_meta.get('cv_r2_mean'),
                        'features': global_features,
                    }
                    if not silent:
                        print(f"✓ GLOBAL model loaded")
                        print(f"  Test R²: {global_meta.get('test_r2'):.4f}")
                        print(f"  Test RMSE: {global_meta.get('test_rmse'):.4f} ppb")
                except Exception as e:
                    print(f"ERROR: Could not load global model: {e}")
                    return None, None, None
            else:
                print(f"ERROR: Global model not found at {global_model_path}")
                return None, None, None
        
        if selected_model is None:
            print(f"ERROR: No model available for predictions")
            return None, None, None
        
        # ────────────────────────────────────────────────────────────────────
        # STEP 7: PREPARE DATA FOR PREDICTION
        # ────────────────────────────────────────────────────────────────────
        if not silent:
            print(f"\n--- PREPARING FOR PREDICTION ---")
        
        # Add features to full GEOS-CF dataset
        geos_data_feat = _add_atmospheric_features(geos_data, spec=spec)
        all_geos = funcs.clean_feature_names(geos_data_feat.copy())
        
        # Determine which features to use
        if use_global and global_features:
            pred_features = global_features
        elif not use_global and model_local:
            pred_features = local_metrics.get('features', sel_feats_clean)
        else:
            print(f"ERROR: Cannot determine prediction features")
            return None, None, None
        
        # Clean feature names
        def _clean_feat(name: str) -> str:
            tmp = pd.DataFrame(columns=[name])
            return funcs.clean_feature_names(tmp).columns[0]
        
        pred_features_clean = [_clean_feat(f) for f in pred_features]
        pred_features_avail = [f for f in pred_features_clean if f in all_geos.columns]
        
        if not silent:
            print(f"✓ Using {len(pred_features_avail)}/{len(pred_features_clean)} features")
        
        # ────────────────────────────────────────────────────────────────────
        # STEP 8: GENERATE PREDICTIONS
        # ────────────────────────────────────────────────────────────────────
        if not silent:
            print(f"\n--- GENERATING PREDICTIONS ---")
        
        all_geos["localised"] = np.nan
        
        # Fill missing values
        pred_df = all_geos[pred_features_avail].copy()
        pred_df = pred_df.ffill().bfill()
        col_medians = pred_df.median()
        pred_df = pred_df.fillna(col_medians)
        
        valid_mask = pred_df.notnull().all(axis=1)
        if not silent:
            print(f"✓ Valid rows: {valid_mask.sum()} / {len(all_geos)}")
        
        if valid_mask.sum() > 0:
            X_pred = pred_df.loc[valid_mask, pred_features_avail].copy()
            model_features = list(selected_model.feature_name_)
            for feat in model_features:
                if feat not in X_pred.columns:
                    X_pred[feat] = 0.0
            X_pred = X_pred[model_features]
            raw_preds = selected_model.predict(X_pred)
            
            # Convert predictions to observation space if needed
            target_mode_pred = selected_metrics.get('target_mode', 'absolute')
            if target_mode_pred == 'ratio' and spec in all_geos.columns:
                raw_conc = all_geos.loc[valid_mask, spec].values
                preds = raw_conc * np.clip(raw_preds, 0.05, 20.0)
            else:
                preds = raw_preds
            
            all_geos.loc[valid_mask, "localised"] = preds
            if not silent:
                print(f"✓ Generated {len(preds)} predictions")
                print(f"  Range: {np.nanmin(preds):.2f} – {np.nanmax(preds):.2f}")
        else:
            print(f"ERROR: No valid rows for prediction")
            return None, None, None
        
        # ────────────────────────────────────────────────────────────────────
        # STEP 9: INTERPOLATE & MERGE
        # ────────────────────────────────────────────────────────────────────
        if interpol:
            n_before = all_geos["localised"].isna().sum()
            all_geos["localised"] = all_geos["localised"].interpolate(
                method="linear", limit=6
            )
            if not silent and n_before > 0:
                filled = n_before - all_geos["localised"].isna().sum()
                print(f"✓ Interpolated {filled} missing values")
        
        # Merge with observations
        result = all_geos.merge(obs_data[["time", "value"]], on="time", how="left")
        
        # Calculate overall AQI
        result = funcs.calculate_overall_aqi(result)
        
        if not silent:
            n_obs = result["value"].notna().sum()
            print(f"\n✓ Final result: {len(result)} rows")
            print(f"  Observations: {n_obs}")
            print(f"  Forecast-only: {len(result) - n_obs}")
        
        # ────────────────────────────────────────────────────────────────────
        # STEP 10: SUMMARY
        # ────────────────────────────────────────────────────────────────────
        if not silent:
            print(f"\n" + "="*70)
            print("FORECAST COMPLETE")
            print("="*70)
            print(f"Location: {loc}")
            print(f"Species: {spec.upper()}")
            print(f"Model source: {selected_metrics.get('source', 'unknown').upper()}")
            if 'cv_r2' in selected_metrics:
                print(f"Model R²: {selected_metrics['cv_r2']:.4f}")
            print(f"Time range: {result['time'].min()} → {result['time'].max()}")
            print(f"="*70 + "\n")
        
        return result, selected_metrics, selected_model
    
    except Exception as exc:
        print(f"ERROR: Forecast generation failed: {exc}")
        traceback.print_exc()
        return None, None, None

