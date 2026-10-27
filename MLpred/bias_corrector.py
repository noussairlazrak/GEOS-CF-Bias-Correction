# -*- coding: utf-8 -*-
"""
bias_corrector.py

Localised forecasts trained on site observations merged with GEOS-CF V1
(historical replay) and V2 (recent analysis + 5-day forecast).

* Pandora sites use the hybrid approach: GEOS-CF × a learned correction factor.
* Other sources (e.g. REMMAQ local monitors) are obs-based: the model predicts
  the local observation directly, with GEOS-CF fields as features.

Site models are chosen among LightGBM, XGBoost and their blend by
time-series CV.
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
    import xgboost as xgb
except ImportError:  # xgboost is optional; LightGBM-only training still works
    xgb = None

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
from MLpred.read_geos_cf import read_geos_cf
from MLpred.s3_manager import S3Manager

# Constants

MODELS_LOCAL_DIR = os.path.join(_REPO_ROOT, "MODELS")
S3_BUCKET        = os.environ.get("GEOS_CF_S3_BUCKET", "smce-geos-cf-public")
S3_MODELS_PREFIX = "snwg_forecast_working_files/MODELS"

PIPELINE_VERSION = 3

HYBRID_OBS_SOURCES = {"pandora"}
HYBRID_TARGETS     = ("ratio", "log_ratio")   # accepted when loading models
OBS_TARGETS        = ("absolute", "log")
HYBRID_TRAIN_TARGETS = ("log_ratio",)
RATIO_CLIP         = (0.05, 20.0)

MIN_FEATURE_COVERAGE = 0.5
MAX_TREES            = 3000
EARLY_STOP_ROUNDS    = 100


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
        payload = {"features": features, "target": target_mode, "metrics": metrics}
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


def _persist_location_metrics(loc_clean: str, spec: str, s3_manager: S3Manager,
                               metrics: dict, silent: bool = False):
    """Record which model (source) served this location's forecast, regardless
    of whether a new model was trained this run."""
    _, _, metrics_path = _model_local_paths(loc_clean, spec)
    os.makedirs(MODELS_LOCAL_DIR, exist_ok=True)
    try:
        with open(metrics_path, "w") as _f:
            json.dump(metrics, _f)
        metrics_s3_key = f"{S3_MODELS_PREFIX}/lgbm_{loc_clean}_{spec}_metrics.json"
        s3_manager.upload_file(metrics_path, metrics_s3_key)
    except Exception as exc:
        if not silent:
            print(f"WARNING: Could not persist location metrics: {exc}")


def _is_hybrid_source(obs_src: str) -> bool:
    """True if *obs_src* is modelled with the hybrid (GEOS-CF × factor) approach."""
    return str(obs_src).lower() in HYBRID_OBS_SOURCES


class BlendRegressor:
    """Average of several regressors trained on the same features and target."""

    def __init__(self, models, names=None):
        self.models = list(models)
        self.names  = list(names) if names else [type(m).__name__ for m in self.models]

    @property
    def feature_name_(self):
        return _model_feature_names(self.models[0])

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)


def _model_feature_names(m) -> list:
    """Feature names a fitted LightGBM / XGBoost / blend model expects, in order."""
    for attr in ("feature_name_", "feature_names_in_"):
        names = getattr(m, attr, None)
        if names is not None:
            return list(names)
    if hasattr(m, "get_booster"):
        return list(m.get_booster().feature_names or [])
    return []


def _clean_name(name: str) -> str:
    return funcs.clean_feature_names(pd.DataFrame(columns=[name])).columns[0]


def _encode_target(obs: np.ndarray, raw: np.ndarray, mode: str) -> np.ndarray:
    """Concentration → model target space."""
    if mode == "absolute":
        return obs.astype(float)
    if mode == "log":
        return np.log1p(np.clip(obs, 0, None))
    ratio = np.clip(obs / np.where(raw > 0, raw, np.nan), *RATIO_CLIP)
    if mode == "ratio":
        return ratio
    if mode == "log_ratio":
        return np.log(ratio)
    raise ValueError(f"Unknown target mode: {mode}")


def _decode_target(pred: np.ndarray, raw: Optional[np.ndarray], mode: str) -> np.ndarray:
    """Model target space → concentration."""
    if mode == "absolute":
        return np.clip(pred, 0, None)
    if mode == "log":
        return np.clip(np.expm1(pred), 0, None)
    if raw is None:
        return np.full(len(pred), np.nan)
    if mode == "ratio":
        return raw * np.clip(pred, *RATIO_CLIP)
    if mode == "log_ratio":
        return raw * np.exp(np.clip(pred, *np.log(RATIO_CLIP)))
    raise ValueError(f"Unknown target mode: {mode}")


def _make_regressor(algo: str, n_estimators: int, n_samples: int,
                    early_stopping: bool = False):
    """Gradient-boosted regressor with defaults tuned for hourly site data."""
    min_leaf = int(np.clip(n_samples // 200, 10, 50))
    if algo == "lgbm_reg":
        return lgb.LGBMRegressor(
            n_estimators=n_estimators, learning_rate=0.03,
            num_leaves=15, max_depth=6, min_child_samples=2 * min_leaf,
            subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
            reg_alpha=1.0, reg_lambda=5.0,
            verbosity=-1, random_state=42,
        )
    if algo == "lgbm":
        return lgb.LGBMRegressor(
            n_estimators=n_estimators, learning_rate=0.03,
            num_leaves=63, max_depth=-1, min_child_samples=min_leaf,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            verbosity=-1, random_state=42,
        )
    if algo == "xgb":
        kw = dict(
            n_estimators=n_estimators, learning_rate=0.03,
            max_depth=7, min_child_weight=min_leaf,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            tree_method="hist", random_state=42, n_jobs=-1,
        )
        if early_stopping:
            kw["early_stopping_rounds"] = EARLY_STOP_ROUNDS
        return xgb.XGBRegressor(**kw)
    raise ValueError(f"Unknown algorithm: {algo}")


def _fit_with_early_stopping(algo: str, X: pd.DataFrame, y: np.ndarray):
    """
    Fit with early stopping on the chronologically last 10% of *X* (never on
    the CV validation fold, which would make CV scores optimistic).
    Returns (model, best_n_trees).
    """
    n_es = max(30, int(len(X) * 0.1))
    if len(X) - n_es < 100:
        m = _make_regressor(algo, 400, len(X))
        m.fit(X, y)
        return m, 400

    X_fit, y_fit = X.iloc[:-n_es], y[:-n_es]
    X_es,  y_es  = X.iloc[-n_es:], y[-n_es:]
    m = _make_regressor(algo, MAX_TREES, len(X_fit), early_stopping=True)
    if algo.startswith("lgbm"):
        m.fit(X_fit, y_fit, eval_set=[(X_es, y_es)],
              callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                         lgb.log_evaluation(-1)])
        best = m.best_iteration_ or MAX_TREES
    else:
        m.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
        best = int(m.best_iteration) + 1
    return m, best


def _score(obs: np.ndarray, pred: np.ndarray):
    ok = np.isfinite(obs) & np.isfinite(pred)
    if ok.sum() < 10:
        return np.nan, np.nan, np.nan
    o, p = obs[ok], pred[ok]
    return (float(np.sqrt(mean_squared_error(o, p))),
            float(r2_score(o, p)),
            float(mean_absolute_error(o, p)))


def _predict_concentration(m, feats, tmode: str, df: pd.DataFrame,
                           spec: str, fill_limit: int = 3) -> np.ndarray:
    """
    Predict concentrations for every row of *df* (time-sorted, clean feature
    names). Short gaps (≤ *fill_limit* h) are filled; anything longer is left
    as NaN for the tree models' native missing-value handling, so V2-only
    columns are not back-filled into the V1 history.
    Rows without a GEOS-CF value for *spec* return NaN.
    """
    cols = _model_feature_names(m) or [_clean_name(f) for f in feats]
    X = pd.DataFrame({c: (df[c] if c in df.columns else np.nan) for c in cols},
                     index=df.index).astype(float)
    X = X.ffill(limit=fill_limit).bfill(limit=fill_limit)

    raw_c = _clean_name(spec)
    raw = (df[raw_c].astype(float).ffill(limit=fill_limit).bfill(limit=fill_limit).values
           if raw_c in df.columns else None)

    conc = _decode_target(m.predict(X), raw, tmode)
    if raw is not None:
        conc = np.where(np.isfinite(raw), conc, np.nan)
    return conc


def _train_site_model(merged: pd.DataFrame, spec: str, hybrid: bool,
                      rmv_out: bool = True, silent: bool = False):
    """
    Train a per-site model, choosing among LightGBM / XGBoost / their blend
    and two target transforms by time-series CV R² (in concentration space).

    * hybrid (Pandora): target = obs / GEOS-CF, as ``ratio`` or ``log_ratio``.
    * obs-based (local sources, e.g. REMMAQ): the model predicts the local
      observation directly, as ``absolute`` or ``log``; GEOS-CF fields are
      just features.

    Returns (model, features, metrics) or (None, None, None).
    """
    raw_col = spec if spec in merged.columns else next(
        (c for c in merged.columns if spec.lower() in c.lower() and c != "value"), None)

    df = merged[np.isfinite(merged["value"].astype(float))].copy()

    if hybrid:
        if raw_col is None:
            print(f"ERROR: Hybrid approach needs a GEOS-CF '{spec}' column — none found")
            return None, None, None
        df = df[(df[raw_col] > 0) & (df["value"] >= 0)]
        target_modes = HYBRID_TRAIN_TARGETS
    else:
        df["value"] = df["value"].clip(lower=0)
        target_modes = OBS_TARGETS

    if rmv_out and len(df) > 100:
        if hybrid:
            lr  = np.log(np.clip(df["value"] / df[raw_col], *RATIO_CLIP)).values.reshape(-1, 1)
            clf = IForest(contamination=0.02, random_state=42)
            clf.fit(lr)
            keep = clf.predict(lr) != 1
        else:
            q1, q3 = df["value"].quantile([0.25, 0.75])
            keep = (df["value"] <= q3 + 10 * (q3 - q1)).values
        if not silent:
            print(f"INFO: Outlier removal: {(~keep).sum()} removed, {keep.sum()} remaining")
        df = df[keep]

    if len(df) < 50:
        print(f"ERROR: Insufficient training data ({len(df)} samples)")
        return None, None, None

    df = df.sort_values("time").reset_index(drop=True)

    skip     = {"time", "location", "lat", "lon", "value", "bias_ratio"}
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in skip]
    coverage = df[num_cols].notna().mean()
    sel_feats = [c for c in num_cols
                 if coverage[c] >= MIN_FEATURE_COVERAGE and df[c].nunique(dropna=True) > 1]
    if not silent:
        dropped = sorted(set(num_cols) - set(sel_feats))
        print(f"INFO: {len(sel_feats)} features used; {len(dropped)} dropped "
              f"(coverage < {MIN_FEATURE_COVERAGE:.0%} or constant)")

    X   = funcs.clean_feature_names(df[sel_feats].astype(float))
    obs = df["value"].values.astype(float)
    raw = df[raw_col].values.astype(float) if raw_col else None
    sel_feats = list(X.columns)

    algos = ["lgbm", "lgbm_reg"] + (["xgb"] if xgb is not None else [])
    if xgb is None and not silent:
        print("WARNING: xgboost not installed — training LightGBM only")

    n_splits = min(5, max(2, len(df) // 500))
    folds    = list(TimeSeriesSplit(n_splits=n_splits).split(X))
    if not silent:
        print(f"INFO: {'Hybrid' if hybrid else 'Obs-based'} training — "
              f"{len(df)} samples, {n_splits}-fold time-series CV, "
              f"algos={algos}, targets={list(target_modes)}")

    def _cv_scores(fold_preds, mode):
        rows = [_score(obs[val], _decode_target(p, raw[val] if raw is not None else None, mode))
                for (_, val), p in zip(folds, fold_preds)]
        return np.array(rows)  # (n_folds, 3) → rmse, r2, mae

    candidates = {}
    for mode in target_modes:
        y = _encode_target(obs, raw, mode)
        per_algo = {}
        for algo in algos:
            fold_preds, iters = [], []
            for tr, val in folds:
                m, n_best = _fit_with_early_stopping(algo, X.iloc[tr], y[tr])
                fold_preds.append(m.predict(X.iloc[val]))
                iters.append(n_best)
            per_algo[algo] = (fold_preds, iters)
            candidates[f"{algo}/{mode}"] = dict(
                algos=[algo], mode=mode, scores=_cv_scores(fold_preds, mode),
                iters={algo: iters})
        if len(per_algo) > 1:
            blend_preds = [np.mean(ps, axis=0) for ps in zip(*(v[0] for v in per_algo.values()))]
            candidates[f"blend/{mode}"] = dict(
                algos=list(per_algo), mode=mode, scores=_cv_scores(blend_preds, mode),
                iters={a: v[1] for a, v in per_algo.items()})

    for name, c in candidates.items():
        c["r2"] = float(np.nanmean(c["scores"][:, 1]))
        if not silent:
            rm, r2, ma = np.nanmean(c["scores"], axis=0)
            print(f"  {name:<18} CV R²={r2:.3f}±{np.nanstd(c['scores'][:, 1]):.3f}  "
                  f"RMSE={rm:.3f}  MAE={ma:.3f}")

    best_name = max(candidates, key=lambda k: candidates[k]["r2"]
                    - (0.005 if k.startswith("blend") else 0.0))
    best = candidates[best_name]
    if not silent:
        for i, (rm, r2, ma) in enumerate(best["scores"], 1):
            print(f"  Fold {i}/{n_splits}: RMSE={rm:.3f}  R²={r2:.3f}  MAE={ma:.3f}")
        print(f"INFO: Selected {best_name} (CV R²={best['r2']:.3f})")

    y = _encode_target(obs, raw, best["mode"])
    models = []
    for algo in best["algos"]:
        n_trees = max(50, int(np.median(best["iters"][algo]) * 1.1))
        m = _make_regressor(algo, n_trees, len(X))
        m.fit(X, y)
        models.append(m)
    model = models[0] if len(models) == 1 else BlendRegressor(models, best["algos"])

    train_pred = _decode_target(model.predict(X), raw, best["mode"])
    cv_rmse, cv_r2, cv_mae = (round(float(v), 3) for v in np.nanmean(best["scores"], axis=0))
    metrics = {
        "CV_RMSE":    cv_rmse,
        "CV_R2":      cv_r2,
        "CV_MAE":     cv_mae,
        "RMSE":       cv_rmse,
        "R2":         cv_r2,
        "MAE":        cv_mae,
        "Train_R2":   round(_score(obs, train_pred)[1], 3),
        "n_folds":    n_splits,
        "n_train":    len(df),
        "trained_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "target":     best["mode"],
        "algorithm":  best_name.split("/")[0],
        "approach":   "hybrid" if hybrid else "obs_based",
        "candidates": {k: round(v["r2"], 3) for k, v in candidates.items()},
        "pipeline_version": PIPELINE_VERSION,
        "obs_start":  str(df["time"].min()),
        "obs_end":    str(df["time"].max()),
        "source":     "trained",
    }
    if not silent:
        print(f"INFO: Final Train R²={metrics['Train_R2']}  CV R²={cv_r2}  "
              f"CV RMSE={cv_rmse}  model={best_name}")
    return model, sel_feats, metrics


# Global model

GLOBAL_TAG = "global"


def _global_paths(spec: str):
    """(model_path, feature_path, metrics_path) of the standard global model."""
    return (os.path.join(MODELS_LOCAL_DIR, f"lgbm_{GLOBAL_TAG}_{spec}_basic.joblib"),
            os.path.join(MODELS_LOCAL_DIR, f"lgbm_{GLOBAL_TAG}_{spec}_features_basic.pkl"),
            os.path.join(MODELS_LOCAL_DIR, f"lgbm_{GLOBAL_TAG}_{spec}_metrics.json"))


def _load_fresh_global(spec: str, model_max_age_days: int):
    """Return (model, metrics, features) for a current-recipe global model
    younger than *model_max_age_days*, else None."""
    model_path, feature_path, _ = _global_paths(spec)
    if not (os.path.exists(model_path) and os.path.exists(feature_path)):
        return None
    payload = pickle.load(open(feature_path, "rb"))
    mets = payload.get("metrics") if isinstance(payload, dict) else None
    if not mets or mets.get("pipeline_version") != PIPELINE_VERSION:
        print(f"Existing global {spec.upper()} model is from an older recipe — retraining.")
        return None
    age_h = (dt.datetime.now().timestamp() - os.path.getmtime(model_path)) / 3600
    if age_h >= model_max_age_days * 24:
        return None
    print(f"Global {spec.upper()} model is fresh ({age_h:.1f}h old) — skipping retraining.")
    return joblib.load(model_path), mets, payload["features"]


def _prepare_global_site_rows(geos_data: pd.DataFrame, obs: pd.DataFrame, spec: str,
                              lon: float, site: str, rmv_out: bool):
    """
    Merge one site's harmonised GEOS-CF (ppbv) with its observations into
    hybrid training rows. Returns a DataFrame, or a string giving the skip reason.
    """
    geos_data = geos_data.sort_values("time").reset_index(drop=True)
    if spec not in geos_data.columns:
        return f"no_{spec}_column"
    med = geos_data[spec].median()
    if not (med > 1e-3):
        raise ValueError(f"GEOS-CF {spec} median={med:.3g} — not in ppbv, "
                         f"pass the data through harmonize_geos_cf()")

    obs = (obs.dropna(subset=["time", "value"])
              .groupby("time", as_index=False)["value"].mean())
    geos_data = _add_atmospheric_features(geos_data, spec=spec, lon=lon)
    merged = (geos_data.drop(columns=[c for c in ("lat", "lon", "location") if c in geos_data.columns])
                       .merge(obs, on="time", how="inner"))
    merged = merged[(merged[spec] > 0) & (merged["value"] >= 0)]
    if len(merged) < 20:
        return "too_few_rows"

    if rmv_out and len(merged) >= 50:
        lr  = np.log(np.clip(merged["value"] / merged[spec], *RATIO_CLIP)).values.reshape(-1, 1)
        clf = IForest(contamination=0.02, random_state=42)
        clf.fit(lr)
        merged = merged[clf.predict(lr) != 1]

    merged["_site"] = site
    float_cols = merged.select_dtypes(include=["float64"]).columns
    merged[float_cols] = merged[float_cols].astype("float32")
    return merged


def _fit_global_model(pool: pd.DataFrame, spec: str, per_site_log: list,
                      extra_metrics: dict, s3_manager: S3Manager, silent: bool = False):
    """
    Fit the global hybrid (log_ratio) model on pooled Pandora sites.

    CV holds out whole sites (GroupKFold), so scores show how the model does
    at a site it has never seen. The out-of-fold R² of every site is stored
    in ``metrics['per_site']``; get_localised_forecast uses it instead of an
    in-sample evaluation when comparing against a site model.
    """
    from sklearn.model_selection import GroupKFold

    pool = pool.sort_values("time").reset_index(drop=True)
    sites = pool["_site"].values
    unique_sites = list(pd.unique(sites))
    if len(unique_sites) < 2:
        print("ERROR: Global model needs at least 2 sites.")
        return None, None, None

    skip     = {"time", "location", "lat", "lon", "value", "bias_ratio", "_site"}
    num_cols = [c for c in pool.select_dtypes(include=[np.number]).columns if c not in skip]
    coverage = pool[num_cols].notna().mean()
    sel      = [c for c in num_cols
                if coverage[c] >= MIN_FEATURE_COVERAGE and pool[c].nunique(dropna=True) > 1]

    X    = funcs.clean_feature_names(pool[sel].astype(float))
    obs  = pool["value"].values.astype(float)
    raw  = pool[spec].values.astype(float)
    mode = "log_ratio"
    y    = _encode_target(obs, raw, mode)
    feats = list(X.columns)

    algos   = ["lgbm", "lgbm_reg"] + (["xgb"] if xgb is not None else [])
    n_folds = min(5, len(unique_sites))
    folds   = list(GroupKFold(n_splits=n_folds).split(X, y, groups=sites))
    print(f"\nPooled: {len(pool):,} rows, {len(unique_sites)} sites, {len(feats)} features. "
          f"{n_folds}-fold site-held-out CV, algos={algos}")

    oof   = {a: np.full(len(pool), np.nan) for a in algos}
    iters = {a: [] for a in algos}
    for i, (tr, val) in enumerate(folds, 1):
        for a in algos:
            m, n_best = _fit_with_early_stopping(a, X.iloc[tr], y[tr])
            oof[a][val] = m.predict(X.iloc[val])
            iters[a].append(n_best)
        if not silent:
            print(f"  fold {i}/{n_folds} done ({len(val):,} held-out rows)")

    candidates = {a: [a] for a in algos}
    if len(algos) > 1:
        oof["blend"] = np.mean([oof[a] for a in algos], axis=0)
        candidates["blend"] = algos

    def _per_site(pred_t):
        conc = _decode_target(pred_t, raw, mode)
        return {s: round(_score(obs[sites == s], conc[sites == s])[1], 3) for s in unique_sites}

    scores = {}
    for name in candidates:
        ps = _per_site(oof[name])
        conc = _decode_target(oof[name], raw, mode)
        rmse, pooled_r2, mae = _score(obs, conc)
        scores[name] = dict(per_site=ps, mean_r2=float(np.nanmean(list(ps.values()))),
                            pooled_r2=pooled_r2, rmse=rmse, mae=mae)
        print(f"  {name:<9} mean site R²={scores[name]['mean_r2']:.3f}  "
              f"pooled R²={pooled_r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}")

    best = max(scores, key=lambda k: scores[k]["mean_r2"] - (0.005 if k == "blend" else 0.0))
    print(f"Selected global model: {best}")

    models = []
    for a in candidates[best]:
        m = _make_regressor(a, max(50, int(np.median(iters[a]) * 1.1)), len(X))
        m.fit(X, y)
        models.append(m)
    model = models[0] if len(models) == 1 else BlendRegressor(models, candidates[best])

    sc = scores[best]
    metrics = {
        "global_CV_RMSE": round(sc["rmse"], 3),
        "global_CV_R2":   round(sc["mean_r2"], 3),
        "global_CV_MAE":  round(sc["mae"], 3),
        "global_pooled_CV_R2": round(sc["pooled_r2"], 3),
        "RMSE":       round(sc["rmse"], 3),
        "R2":         round(sc["mean_r2"], 3),
        "MAE":        round(sc["mae"], 3),
        "Train_R2":   round(_score(obs, _decode_target(model.predict(X), raw, mode))[1], 3),
        "per_site":   sc["per_site"],
        "site_summary": per_site_log,
        "candidates": {k: round(v["mean_r2"], 3) for k, v in scores.items()},
        "n_train":    int(len(pool)),
        "n_sites":    len(unique_sites),
        "target":     mode,
        "algorithm":  best,
        "approach":   "hybrid",
        "spec":       spec,
        "source":     "global",
        "pipeline_version": PIPELINE_VERSION,
        "trained_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        **extra_metrics,
    }
    print(f"Global model — CV mean site R²={metrics['R2']}  Train R²={metrics['Train_R2']}")

    import json as _json
    model_path, feature_path, metrics_path = _global_paths(spec)
    os.makedirs(MODELS_LOCAL_DIR, exist_ok=True)
    try:
        joblib.dump(model, model_path)
        pickle.dump({"features": feats, "target": mode, "metrics": metrics}, open(feature_path, "wb"))
        with open(metrics_path, "w") as _f:
            _json.dump(metrics, _f, indent=2)
        print(f"Global model saved → {model_path}")
    except Exception as exc:
        print(f"WARNING: Could not save global model locally: {exc}")

    try:
        for path in (model_path, feature_path, metrics_path):
            s3_manager.upload_file(path, f"{S3_MODELS_PREFIX}/{os.path.basename(path)}")
        print(f"Global model uploaded to S3.")
    except Exception as exc:
        print(f"WARNING: Could not upload global model to S3: {exc}")

    return model, metrics, feats


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
    Train a single **global** hybrid model pooled across Pandora locations.

    Global models are hybrid (GEOS-CF × learned factor), so only locations
    with a hybrid observation source (Pandora) are used; others are skipped.

    Parameters
    ----------
    locations : list of dict
        Each dict must contain at minimum:
          ``loc``      – human-readable site name (as passed to
                         ``get_localised_forecast``, so per-site CV scores match)
          ``lat``      – latitude
          ``lon``      – longitude
          ``obs_src``  – observation source (``'pandora'``)
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
        Apply Isolation-Forest outlier removal (on log obs/model) per site.
    obs_cache_hours : int
        Pandora cache staleness threshold.
    force_obs_refresh : bool
        Delete and re-download observation caches.
    model_max_age_days : int
        If a current-recipe global model younger than this exists, skip retraining.
    silent : bool
        Suppress verbose output.
    s3_manager : S3Manager, optional
        Pre-configured S3Manager.

    Returns
    -------
    model : LGBMRegressor, XGBRegressor, BlendRegressor or None
    metrics : dict or None
        Keys include ``global_CV_R2`` (mean held-out site R²), ``per_site``
        ({site: out-of-fold R²}), ``n_train``, ``n_sites``, ``trained_at``, ``target``.
    feature_names : list[str] or None
    """
    if st is None:
        st = dt.datetime(2018, 1, 1)
    if ed is None:
        ed = dt.datetime.today()
    if s3_manager is None:
        s3_manager = S3Manager(bucket_name=S3_BUCKET)

    fresh = _load_fresh_global(spec, model_max_age_days)
    if fresh is not None:
        return fresh

    all_frames   = []
    per_site_log = []

    print(f"\n{'='*60}")
    print(f"Training GLOBAL {spec.upper()} model  ({len(locations)} locations)")
    print(f"{'='*60}")

    for loc_cfg in locations:
        loc      = loc_cfg["loc"]
        lat      = float(loc_cfg["lat"])
        lon      = float(loc_cfg["lon"])
        obs_src  = loc_cfg.get("obs_src", "pandora")

        if not _is_hybrid_source(obs_src):
            print(f"  [{loc}] skipped — obs_src='{obs_src}' is obs-based, not hybrid.")
            per_site_log.append({"loc": loc, "status": "not_hybrid_source", "n_rows": 0})
            continue

        print(f"\n  [{loc}]  lat={lat:.3f}  lon={lon:.3f}  src={obs_src}")
        try:
            obs_data = _load_observations(
                obs_src=obs_src, obs_url=loc_cfg.get("obs_url", ""), openaq_id=None,
                spec=spec, lat=lat, lon=lon, mod_src="s3", unit=unit,
                time_col=loc_cfg.get("time_col", "time"),
                date_fmt=loc_cfg.get("date_fmt", "%Y-%m-%d %H:%M"),
                obs_val_col=loc_cfg.get("obs_val_col", "value"),
                lat_col=loc_cfg.get("lat_col"), lon_col=loc_cfg.get("lon_col"),
                rmv_out=False, cache_hours=obs_cache_hours,
                force_refresh=force_obs_refresh, silent=silent,
            )
            if obs_data is None or obs_data.empty:
                print(f"    WARNING: No observations — skipping {loc}.")
                per_site_log.append({"loc": loc, "status": "no_obs", "n_rows": 0})
                continue

            frames_g = []
            for ver, end in ((1, ed), (2, datetime.now() + timedelta(days=5))):
                gv = read_geos_cf(lon=lon, lat=lat, start=st, end=end, version=ver, verbose=False)
                if gv is not None and not gv.empty:
                    gv["time"] = _normalize_time(gv["time"])
                    frames_g.append(gv)
            if not frames_g:
                print(f"    WARNING: No GEOS-CF data — skipping {loc}.")
                per_site_log.append({"loc": loc, "status": "no_geos_cf", "n_rows": 0})
                continue
            geos_data = (pd.concat(frames_g, ignore_index=True)
                           .drop_duplicates(subset=["time"], keep="first"))

            rows = _prepare_global_site_rows(geos_data, obs_data[["time", "value"]],
                                             spec, lon, loc, rmv_out)
            if isinstance(rows, str):
                print(f"    WARNING: {rows} — skipping {loc}.")
                per_site_log.append({"loc": loc, "status": rows, "n_rows": 0})
                continue
            all_frames.append(rows)
            per_site_log.append({"loc": loc, "status": "ok", "n_rows": len(rows)})
            print(f"    OK — {len(rows)} training rows.")

        except Exception as exc:
            import traceback as _tb
            print(f"    ERROR loading {loc}: {exc}")
            if not silent:
                _tb.print_exc()
            per_site_log.append({"loc": loc, "status": f"error: {exc}", "n_rows": 0})

    if not all_frames:
        print("ERROR: No usable data collected for global model training.")
        return None, None, None

    pool = pd.concat(all_frames, ignore_index=True)
    return _fit_global_model(pool, spec, per_site_log, {}, s3_manager, silent=silent)


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
    Train the global hybrid model using **only locally cached** Pandora
    observation and GEOS-CF files — no remote downloads.

    File discovery
    --------------
    Observations  : ``<obs_dir>/<Site>_<spec>.csv`` (the Pandora cache)
    GEOS-CF       : ``<geos_cf_dir>/loc_<lat>_<lon>_v{1,2}.csv``

    The GEOS-CF cache stores raw values (mol/mol, V1 column names), so it is
    passed through ``harmonize_geos_cf`` — the same step ``read_geos_cf``
    applies — before training. Skipping it was what broke the old global model.

    The merged training dataset is saved as a Parquet file alongside the model
    (``MODELS/global_<spec>_pool.parquet``) for future analysis or re-use.

    Returns
    -------
    model : LGBMRegressor, XGBRegressor, BlendRegressor or None
    metrics : dict or None
    feature_names : list[str] or None
    """
    import glob
    from MLpred.read_geos_cf import harmonize_geos_cf

    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO     = os.path.dirname(_THIS_DIR)

    if obs_dir is None:
        obs_dir = OBS_CACHE_DIR
    if geos_cf_dir is None:
        geos_cf_dir = os.path.join(_REPO, "GEOS_CF")
    if s3_manager is None:
        s3_manager = S3Manager(bucket_name=S3_BUCKET)

    pool_path = os.path.join(MODELS_LOCAL_DIR, f"global_{spec}_pool.parquet")

    fresh = _load_fresh_global(spec, model_max_age_days)
    if fresh is not None:
        return fresh

    pattern   = os.path.join(obs_dir, f"*_{spec.lower()}.csv")
    obs_files = sorted(glob.glob(pattern))
    if not obs_files:
        print(f"No OBS files found: {pattern}")
        return None, None, None

    print(f"Training global {spec.upper()} model — {len(obs_files)} OBS files found.")

    def _coord_to_str(val: float) -> str:
        return f"{val:.6f}".replace(".", "_").replace("-", "m")

    def _find_geos_cf(lat: float, lon: float) -> pd.DataFrame:
        lat_s, lon_s = _coord_to_str(lat), _coord_to_str(lon)
        frames_g = []
        for ver in (1, 2):
            fpath = os.path.join(geos_cf_dir, f"loc_{lat_s}_{lon_s}_v{ver}.csv")
            if os.path.exists(fpath):
                try:
                    df = pd.read_csv(fpath)
                    df["time"] = _normalize_time(df["time"])
                    frames_g.append(harmonize_geos_cf(df))
                except Exception:
                    pass
        if not frames_g:
            return pd.DataFrame()
        return (pd.concat(frames_g, ignore_index=True)
                  .drop_duplicates(subset=["time"], keep="first"))

    all_frames   = []
    per_site_log = []

    for obs_path in obs_files:
        site_name = os.path.basename(obs_path).replace(f"_{spec.lower()}.csv", "")
        try:
            obs = pd.read_csv(obs_path)
            if "value" not in obs.columns:
                for alt in (spec, spec.lower(), "concentration", "val", "obs"):
                    if alt in obs.columns:
                        obs = obs.rename(columns={alt: "value"})
                        break
            if "value" not in obs.columns:
                per_site_log.append({"loc": site_name, "status": "no_value_col", "n_rows": 0})
                continue
            obs["time"] = _normalize_time(obs["time"])
            obs = obs.dropna(subset=["time", "value"])
            if st is not None:
                obs = obs[obs["time"] >= pd.Timestamp(st)]
            if ed is not None:
                obs = obs[obs["time"] <= pd.Timestamp(ed)]
            if obs.empty:
                per_site_log.append({"loc": site_name, "status": "no_obs", "n_rows": 0})
                continue

            lat = lon = None
            for c in ("lat", "latitude", "Latitude"):
                if c in obs.columns and obs[c].notna().any():
                    lat = float(obs[c].dropna().iloc[0])
                    break
            for c in ("lon", "longitude", "Longitude"):
                if c in obs.columns and obs[c].notna().any():
                    lon = float(obs[c].dropna().iloc[0])
                    break
            if lat is None or lon is None:
                per_site_log.append({"loc": site_name, "status": "no_coords", "n_rows": 0})
                continue

            geos_data = _find_geos_cf(lat, lon)
            if geos_data.empty:
                per_site_log.append({"loc": site_name, "status": "no_geos_cf", "n_rows": 0})
                continue

            rows = _prepare_global_site_rows(geos_data, obs[["time", "value"]],
                                             spec, lon, site_name, rmv_out)
            del geos_data, obs
            if isinstance(rows, str) or len(rows) < min_overlap_rows:
                status = rows if isinstance(rows, str) else "too_few_rows"
                per_site_log.append({"loc": site_name, "status": status, "n_rows": 0})
                continue
            all_frames.append(rows)
            per_site_log.append({"loc": site_name, "status": "ok", "n_rows": len(rows)})

        except Exception as exc:
            per_site_log.append({"loc": site_name, "status": f"error: {exc}", "n_rows": 0})

    n_ok = sum(1 for s in per_site_log if s["status"] == "ok")
    print(f"{n_ok} sites usable, {len(per_site_log) - n_ok} skipped.")
    if not all_frames:
        print("No usable data collected — global model not trained.")
        return None, None, None

    pool = pd.concat(all_frames, ignore_index=True)
    del all_frames
    gc.collect()

    os.makedirs(MODELS_LOCAL_DIR, exist_ok=True)
    try:
        pool.to_parquet(pool_path, index=False)
        if not silent:
            print(f"Pooled dataset saved → {pool_path}")
    except Exception as exc:
        if not silent:
            print(f"WARNING: Could not save pooled dataset: {exc}")

    extra = {"source": "global_local", "obs_dir": obs_dir,
             "geos_cf_dir": geos_cf_dir, "pool_path": pool_path}
    return _fit_global_model(pool, spec, per_site_log, extra, s3_manager, silent=silent)



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
        df["time"] = _normalize_time(df["time"])
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
    df["time"] = _normalize_time(df["time"])
    return df

def _format_metrics_summary(metrics: dict) -> str:
    """One-line, human-readable summary of a model metrics dict for logging."""
    if not metrics:
        return "no metrics available"
    keys = ("source", "approach", "algorithm", "target", "R2", "RMSE", "MAE",
            "CV_R2", "CV_RMSE", "CV_MAE", "Train_R2", "n_train", "trained_at")
    bits = [f"{k}={metrics[k]}" for k in keys if metrics.get(k) is not None]
    return ", ".join(bits) if bits else "no metrics available"


def _normalize_time(series):
    """
    Convert mixed datetime strings to UTC, floor to hour,
    then remove timezone so all merges use datetime64[ns].
    """
    return (
        pd.to_datetime(
            series,
            utc=True,
            format="mixed",
            errors="coerce"
        )
        .dt.floor("h")
        .dt.tz_localize(None)
    )

def _add_atmospheric_features(df: pd.DataFrame, spec: str = "no2",
                              lon: Optional[float] = None) -> pd.DataFrame:
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
        ``lst_hour``, ``lst_weekday``, ``lst_is_weekend`` – same in local
                         solar time (only when *lon* is given), so traffic
                         rush hours and weekends line up with local habits

    Atmospheric proxies
        ``pbl_proxy``  – log(PBLH) if a PBL-height column is present; captures
                         mixing-layer dilution effect on surface concentrations.
        ``t2m_delta``  – T2M − T850 (lapse-rate proxy for convective mixing)
                         only added when both columns exist.
        ``wind_speed``, ``wind_dir_sin``, ``wind_dir_cos`` – from u/v
        ``ventilation`` – wind speed × PBLH (dispersion capacity)
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

    if lon is not None:
        lst = t + pd.to_timedelta(lon / 15.0, unit="h")
        out["lst_hour"]       = (lst.dt.hour + lst.dt.minute / 60.0).astype(float)
        out["lst_weekday"]    = lst.dt.weekday.astype(float)
        out["lst_is_weekend"] = (lst.dt.weekday >= 5).astype(float)

    pbl_candidates = [c for c in out.columns
                      if "pbl" in c.lower() or "pbh" in c.lower() or "kpbl" in c.lower()]
    pbl_col = pbl_candidates[0] if pbl_candidates else None
    if pbl_col:
        out["pbl_proxy"] = np.log1p(out[pbl_col].clip(lower=0))

    u_col = next((c for c in ("u", "u10m", "u10") if c in out.columns), None)
    v_col = next((c for c in ("v", "v10m", "v10") if c in out.columns), None)
    if u_col and v_col:
        out["wind_speed"]   = np.hypot(out[u_col], out[v_col])
        wd = np.arctan2(out[v_col], out[u_col])
        out["wind_dir_sin"] = np.sin(wd)
        out["wind_dir_cos"] = np.cos(wd)
        if pbl_col:
            out["ventilation"] = out["wind_speed"] * out[pbl_col].clip(lower=0)

    # surface–850 hPa temperature difference
    t2m_cols  = [c for c in out.columns if "t2m"  in c.lower() or "ts"   in c.lower()]
    t850_cols = [c for c in out.columns if "t850" in c.lower() or "t_850" in c.lower()]
    if t2m_cols and t850_cols:
        out["t2m_delta"] = out[t2m_cols[0]] - out[t850_cols[0]]

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
    retrain_min_new_obs_days: int = 60,
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
       c. Train a new site model if none is found (or ``force_retrain``):
          LightGBM, XGBoost or their blend, picked by time-series CV.
       Global models are hybrid (Pandora-trained) and only considered for
       Pandora sites; other sources use obs-based site models only.

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
    retrain_min_new_obs_days : int
        A current-recipe site model is only retrained once the observations
        extend at least this many days past the data it was trained on
        (default: 60). Until then the best pre-trained model is used, whatever
        its age or R². Models from an older recipe, or built for the other
        approach (hybrid vs obs-based), are still retrained once.
    model_max_age_days : int
        Kept for backward compatibility; site-model retraining is now driven
        by new observations (``retrain_min_new_obs_days``), not model age.
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

        obs_data = (obs_data.dropna(subset=["time", "value"])
                            .groupby("time", as_index=False)["value"].mean())

        if not silent:
            print(f"INFO: {len(obs_data)} observations "
                  f"({obs_data['time'].min()} → {obs_data['time'].max()})")

        # GEOS-CF V1
        if not silent:
            print(f"INFO: Loading GEOS-CF V1 (replay)…")
        geos_v1 = read_geos_cf(lon=lon, lat=lat, start=st, end=ed,
                                       version=1, verbose=not silent)

        # GEOS-CF V2 
        if not silent:
            print(f"INFO: Loading GEOS-CF V2 (analysis + forecast)…")
        geos_v2 = read_geos_cf(
            lon=lon, lat=lat,
            start=st,
            end=datetime.now() + timedelta(days=5),
            version=2, verbose=not silent,
        )

        # Combine V1 + V2 
        frames = []
        if geos_v1 is not None and not geos_v1.empty:
            geos_v1["time"] = _normalize_time(geos_v1["time"])
            frames.append(geos_v1)
            if not silent:
                print(f"INFO: V1 → {len(geos_v1)} rows "
                      f"({geos_v1['time'].min()} → {geos_v1['time'].max()})")

        if geos_v2 is not None and not geos_v2.empty:
            geos_v2["time"] = _normalize_time(geos_v2["time"])
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

        # compute on the full series, not the sparse obs subset, so lag
        # features mean the same thing at train time and predict time
        geos_data_feat = _add_atmospheric_features(geos_data, spec=spec, lon=lon)
        all_geos = funcs.clean_feature_names(geos_data_feat.copy())

        hybrid   = _is_hybrid_source(obs_src)
        approach = "hybrid" if hybrid else "obs_based"
        if not silent:
            print(f"INFO: Approach = {approach} "
                  f"({'GEOS-CF × learned correction factor' if hybrid else 'direct model of local observations'})")

        R2_MIN_THRESHOLD = 0.50
        import json as _json

        model, sel_feats, metrics = None, None, {}
        model_path, feature_path, metrics_path = _model_local_paths(loc_clean, spec)

        # best candidate below threshold, used if nothing clears the bar
        _best_fallback = {"model": None, "feats": None, "metrics": None, "r2": -np.inf}
        _legacy = {}

        def _is_legacy(m, feats, mets, label):
            if (mets or {}).get("pipeline_version") == PIPELINE_VERSION:
                return False
            _legacy.setdefault("entry", (m, feats, mets))
            if not silent:
                print(f"INFO: {label} is from an older training recipe "
                      f"(pipeline_version={(mets or {}).get('pipeline_version')}, "
                      f"current={PIPELINE_VERSION}) — not scored, will retrain.")
            return True

        def _consider_fallback(m, feats, mets, r2):
            if m is not None and r2 is not None and r2 > _best_fallback["r2"]:
                _best_fallback.update(model=m, feats=feats, metrics=mets, r2=r2)

        def _compatible(tmode, label):
            ok = (tmode in HYBRID_TARGETS) == hybrid
            if not ok and not silent:
                print(f"INFO: Ignoring {label} — target '{tmode}' does not match the "
                      f"{approach} approach used for obs_src='{obs_src}'.")
            return ok

        _obs_end = obs_data["time"].max()

        def _new_obs_days(mets):
            """Days of observations beyond what the model was trained on.
            Models that predate the ``obs_end`` field fall back to their
            training date (no obs after it can have been used)."""
            ref = (mets or {}).get("obs_end") or (mets or {}).get("trained_at")
            try:
                return max(0, (_obs_end - pd.Timestamp(ref)).days)
            except (TypeError, ValueError):
                return None

        def _needs_new_data_retrain(mets):
            d = _new_obs_days(mets)
            return d is None or d >= retrain_min_new_obs_days

        def _load_and_eval(mdl_path, feat_path, met_path=None, label="model"):
            """
            Load a model + feature payload from disk.
            Returns (model, feats, target_mode, r2, full_metrics_dict)
            or (None, None, None, None, None) on failure.
            """
            try:
                m   = joblib.load(mdl_path)
                pay = pickle.load(open(feat_path, "rb"))
                feats  = pay["features"] if isinstance(pay, dict) else pay
                tmode  = pay.get("target", "absolute") if isinstance(pay, dict) else "absolute"
                mets   = {"source": label, "target": tmode}
                _pm = pay.get("metrics") if isinstance(pay, dict) else None
                if not _pm and met_path and os.path.exists(met_path):
                    try:
                        with open(met_path) as _ff:
                            _pm = _json.load(_ff)
                    except Exception:
                        _pm = None
                if _pm:
                    mets.update({k: _pm.get(k) for k in
                                 ("RMSE", "R2", "MAE", "CV_R2", "n_train", "trained_at",
                                  "algorithm", "approach", "pipeline_version",
                                  "obs_end", "per_site",
                                  "global_CV_RMSE", "global_CV_R2")
                                 if _pm.get(k) is not None})
                return m, feats, tmode, mets.get("R2"), mets
            except Exception as _le:
                if not silent:
                    print(f"WARNING: Could not load {label} from {mdl_path}: {_le}")
                return None, None, None, None, None

        _obs_by_time = obs_data.drop_duplicates("time").set_index("time")["value"]

        def _eval_on_obs(m, feats, tmode):
            """
            Evaluate model *m* against the obs/GEOS overlap. Returns (rmse, r2, mae).
            Note: in-sample for models that were trained on this site.
            """
            try:
                conc = _predict_concentration(m, feats, tmode, all_geos, spec)
                obs_v = _obs_by_time.reindex(all_geos["time"].values).values.astype(float)
                rmse, r2, mae = _score(obs_v, conc)
                if np.isnan(r2):
                    return None, None, None
                return rmse, r2, mae
            except Exception as _ee:
                if not silent:
                    print(f"WARNING: Eval failed: {_ee}")
                return None, None, None

        def _score_global(gm, gf, gt, gmets, label):
            """
            R² of a global model for this site, comparable to a site model's
            CV R²: the site's out-of-fold score from global training if it
            was in the pool, else a live eval (honest, as the site is unseen).
            Returns (r2, metrics), or (None, None) for an old-recipe global
            model (trained on raw mol/mol GEOS-CF — its predictions are wrong).
            """
            if (gmets or {}).get("pipeline_version") != PIPELINE_VERSION:
                if not silent:
                    print(f"INFO: {label} is from an older recipe — ignored. "
                          f"Retrain it with train_global_model_from_local().")
                return None, None
            mets = {k: v for k, v in gmets.items() if k != "per_site"}
            site_r2 = (gmets.get("per_site") or {}).get(loc)
            if site_r2 is not None and np.isfinite(site_r2):
                mets["R2"] = site_r2
                if not silent:
                    print(f"INFO: {label} — out-of-fold R² for {loc} = {site_r2:.3f}")
                return site_r2, mets
            rmse, r2, mae = _eval_on_obs(gm, gf, gt)
            if r2 is None:
                return None, None
            mets.update({"RMSE": round(rmse, 3), "R2": round(r2, 3), "MAE": round(mae, 3)})
            if not silent:
                print(f"INFO: {label} — {loc} not in its training pool, live eval R²={r2:.3f}")
            return r2, mets

        skip_retrain = False

        # step 1: local model
        local_r2 = None
        if not force_retrain and os.path.exists(model_path) and os.path.exists(feature_path):
            if not silent:
                age_h = (dt.datetime.now().timestamp() - os.path.getmtime(model_path)) / 3600
                print(f"INFO: Found local model ({age_h:.1f}h old) → {model_path}")
            _m, _f, _t, _r2_stored, _mets = _load_and_eval(
                model_path, feature_path, metrics_path, label="local")
            if (_m is not None and _compatible(_t, "local model")
                    and not _is_legacy(_m, _f, _mets, "Local model")):
                # Evaluate live if stored R² is missing
                if _r2_stored is None:
                    _rmse, _r2_live, _mae = _eval_on_obs(_m, _f, _t)
                    if _r2_live is not None:
                        _mets.update({"RMSE": round(_rmse, 3),
                                      "R2":   round(_r2_live, 3),
                                      "MAE":  round(_mae, 3)})
                        local_r2 = _r2_live
                    if not silent:
                        print(f"INFO: Local model live eval — R²={_r2_live}  RMSE={_rmse}")
                else:
                    local_r2 = _r2_stored
                    if not silent:
                        print(f"INFO: Local model stored R²={local_r2} "
                              f"({_mets.get('algorithm', 'lgbm')}, target={_t})")

                _new_days = _new_obs_days(_mets)
                skip_retrain = not _needs_new_data_retrain(_mets)
                if not silent:
                    print(f"INFO: {_new_days if _new_days is not None else 'unknown'} days of "
                          f"new obs since this model's training data "
                          + ("— below the retrain threshold, keeping pre-trained models."
                             if skip_retrain else
                             f"(≥ {retrain_min_new_obs_days}) — will retrain."))

                if local_r2 is not None and local_r2 >= R2_MIN_THRESHOLD and skip_retrain:
                    model, sel_feats, metrics = _m, _f, _mets
                    if not silent:
                        print(f"INFO: Local model accepted (R²={local_r2:.3f} ≥ {R2_MIN_THRESHOLD}).")
                else:
                    _consider_fallback(_m, _f, _mets, local_r2)
                    if not silent and (local_r2 is None or local_r2 < R2_MIN_THRESHOLD):
                        print(f"WARNING: Local model below threshold "
                              f"(R²={local_r2 if local_r2 is not None else 'unknown'} "
                              f"< {R2_MIN_THRESHOLD}).")

        elif not force_retrain and not silent:
            print(f"INFO: No local pre-trained model found at {model_path}")

        GLOBAL_TAG          = "global"
        # Preferred path: species-specific named model e.g. global_no2_model.joblib
        _glob_named_path    = os.path.join(MODELS_LOCAL_DIR, f"global_{spec}_model.joblib")
        _glob_named_feats   = os.path.join(MODELS_LOCAL_DIR, f"global_{spec}_model_features.pkl")
        _glob_named_metrics = os.path.join(MODELS_LOCAL_DIR, f"global_{spec}_model_metrics.json")
        # Fallback: standard naming from train_global_model()
        _glob_std_path      = os.path.join(MODELS_LOCAL_DIR, f"lgbm_{GLOBAL_TAG}_{spec}_basic.joblib")
        _glob_std_feats     = os.path.join(MODELS_LOCAL_DIR, f"lgbm_{GLOBAL_TAG}_{spec}_features_basic.pkl")
        _glob_std_metrics   = os.path.join(MODELS_LOCAL_DIR, f"lgbm_{GLOBAL_TAG}_{spec}_metrics.json")

        # Pick whichever global file exists (named preferred)
        _gpath = None
        if not hybrid:
            if not silent:
                print(f"INFO: Skipping global models (hybrid, Pandora-trained) for obs-based source.")
        elif os.path.exists(_glob_named_path) and os.path.exists(_glob_named_feats):
            _gpath, _gfpath, _gmpath = _glob_named_path, _glob_named_feats, _glob_named_metrics
            _glabel = f"global_pretrained ({os.path.basename(_glob_named_path)})"
        elif os.path.exists(_glob_std_path) and os.path.exists(_glob_std_feats):
            _gpath, _gfpath, _gmpath = _glob_std_path, _glob_std_feats, _glob_std_metrics
            _glabel = f"global_standard ({os.path.basename(_glob_std_path)})"

        global_model     = None
        global_sel_feats = None
        global_r2        = None
        global_metrics   = {}

        if _gpath is not None:
            if not silent:
                print(f"INFO: Found global model → {_gpath}")
            _gm, _gf, _gt, _gr2_stored, _gmets = _load_and_eval(
                _gpath, _gfpath, _gmpath, label=_glabel)
            if _gm is not None and _compatible(_gt, "global model"):
                global_r2, _gscored = _score_global(_gm, _gf, _gt, _gmets, "Global model")
                if _gscored is not None:
                    global_model     = _gm
                    global_sel_feats = _gf
                    global_metrics   = _gscored

        # Decide: switch to global?
        if global_model is not None:
            _consider_fallback(global_model, global_sel_feats, global_metrics, global_r2)
            local_ok  = model is not None and local_r2 is not None and local_r2 >= R2_MIN_THRESHOLD
            global_ok = global_r2 is not None and global_r2 >= R2_MIN_THRESHOLD

            if global_ok and (not local_ok or global_r2 > (local_r2 or -np.inf)):
                reason = (
                    "no local model" if model is None
                    else f"global R²={global_r2:.3f} > local R²={local_r2:.3f}"
                )
                if not silent:
                    print(f"INFO: Using global model ({reason}).")
                model      = global_model
                sel_feats  = global_sel_feats
                metrics    = global_metrics
            elif local_ok:
                if not silent:
                    print(f"INFO: Keeping local model "
                          f"(local R²={local_r2:.3f} ≥ {R2_MIN_THRESHOLD} "
                          f"and ≥ global R²={global_r2}).")
            else:
                model = None
                if not silent:
                    print(f"INFO: Local R²={local_r2} and global R²={global_r2} both "
                          f"< {R2_MIN_THRESHOLD} — continuing search.")

        if model is None and not force_retrain and not skip_retrain:
            if not silent:
                print(f"INFO: Checking S3 for site-specific model…")
            _s3m, _s3pay, _s3mets = _try_load_model_from_s3(
                loc_clean, spec, s3_manager, silent=silent)
            if _s3m is not None:
                _s3f  = _s3pay["features"] if isinstance(_s3pay, dict) else _s3pay
                _s3t  = _s3pay.get("target", "absolute") if isinstance(_s3pay, dict) else "absolute"
                _s3own = _s3pay.get("metrics") if isinstance(_s3pay, dict) else None
                if (_compatible(_s3t, "S3 site model")
                        and not _is_legacy(_s3m, _s3f,
                                           dict(_s3own or _s3mets or {}, source="s3", target=_s3t),
                                           "S3 site model")):
                    if _s3own and _s3own.get("R2") is not None:
                        _s3_metrics = dict(_s3own, source="s3", target=_s3t)
                        _s3r2 = _s3own["R2"]
                    else:
                        _s3rm, _s3r2, _s3ma = _eval_on_obs(_s3m, _s3f, _s3t)
                        _s3_metrics = {"source": "s3", "target": _s3t,
                                       "RMSE": round(_s3rm, 3) if _s3rm is not None else None,
                                       "R2":   round(_s3r2, 3) if _s3r2 is not None else None,
                                       "MAE":  round(_s3ma, 3) if _s3ma is not None else None}
                        if _s3mets:
                            _s3_metrics.update({k: _s3mets.get(k)
                                                for k in ("n_train", "trained_at")})
                    if not silent:
                        print(f"INFO: S3 site model — R²={_s3r2 if _s3r2 is None else round(_s3r2, 3)}")
                    skip_retrain = not _needs_new_data_retrain(_s3_metrics)
                    if _s3r2 is not None and _s3r2 >= R2_MIN_THRESHOLD and skip_retrain:
                        model, sel_feats, metrics = _s3m, _s3f, _s3_metrics
                        if not silent:
                            print(f"INFO: S3 site model accepted (R²={_s3r2:.3f}).")
                    else:
                        _consider_fallback(_s3m, _s3f, _s3_metrics, _s3r2)
                        if not silent:
                            print(f"WARNING: S3 site model below threshold (R²={_s3r2 if _s3r2 is None else round(_s3r2, 3)}).")

        if model is None and not force_retrain and hybrid:
            if not silent:
                print(f"INFO: Checking S3 for global model…")
            os.makedirs(MODELS_LOCAL_DIR, exist_ok=True)
            for _s3_key, _local_dst, _s3_fkey, _local_fdst in [
                (f"{S3_MODELS_PREFIX}/global_{spec}_model.joblib",
                 _glob_named_path,
                 f"{S3_MODELS_PREFIX}/global_{spec}_model_features.pkl",
                 _glob_named_feats),
                (f"{S3_MODELS_PREFIX}/lgbm_{GLOBAL_TAG}_{spec}_basic.joblib",
                 _glob_std_path,
                 f"{S3_MODELS_PREFIX}/lgbm_{GLOBAL_TAG}_{spec}_features_basic.pkl",
                 _glob_std_feats),
            ]:
                if model is not None:
                    break
                if global_model is not None and _local_dst == _gpath:
                    continue
                try:
                    _got_m = s3_manager.download_file(_s3_key,  _local_dst)
                    _got_f = s3_manager.download_file(_s3_fkey, _local_fdst)
                    if _got_m and _got_f:
                        _gm2, _gf2, _gt2, _, _g2mets = _load_and_eval(
                            _local_dst, _local_fdst, label="s3_global")
                        if _gm2 is not None and _compatible(_gt2, "S3 global model"):
                            _gr2v, _g4_metrics = _score_global(_gm2, _gf2, _gt2, _g2mets,
                                                               "S3 global model")
                            if _g4_metrics is None:
                                continue
                            _g4_metrics["source"] = "s3_global"
                            _consider_fallback(_gm2, _gf2, _g4_metrics, _gr2v)
                            if _gr2v is not None and _gr2v >= R2_MIN_THRESHOLD:
                                model, sel_feats, metrics = _gm2, _gf2, _g4_metrics
                            elif not silent:
                                print(f"WARNING: S3 global model below threshold (R²={_gr2v}).")
                except Exception:
                    pass

        # step 5: train new model, last resort
        if model is None and not skip_retrain:
            if not silent:
                print(f"INFO: Training new {approach} model for {loc}…")

            merged_train = (
                geos_data_feat.merge(obs_data[["time", "value"]], on="time", how="inner")
                              .dropna(subset=["value"])
            )
            if not silent:
                print(f"INFO: Training samples after merge: {len(merged_train)}")

            _new_m, _new_f, _new_mets = _train_site_model(
                merged_train, spec, hybrid=hybrid, rmv_out=rmv_out, silent=silent)

            if _new_m is not None:
                cv_r2 = _new_mets["CV_R2"]
                _save_model(_new_m, _new_f, loc_clean, spec, s3_manager, silent=silent,
                            target_mode=_new_mets["target"], metrics=_new_mets)
                if cv_r2 >= R2_MIN_THRESHOLD:
                    model, sel_feats, metrics = _new_m, _new_f, _new_mets
                else:
                    if not silent:
                        print(f"WARNING: New model CV R²={cv_r2:.3f} < {R2_MIN_THRESHOLD} "
                              f"— comparing with other candidates.")
                    _consider_fallback(_new_m, _new_f, _new_mets, cv_r2)

        # nothing cleared the threshold, use the best candidate seen
        if model is None:
            if _best_fallback["model"] is None and _legacy:
                _lm, _lf, _lmets = _legacy["entry"]
                if not silent:
                    print(f"WARNING: No current-recipe model available — serving legacy model.")
                model, sel_feats, metrics = _lm, _lf, _lmets
            elif _best_fallback["model"] is not None:
                if not silent:
                    print(f"INFO: No candidate met R²≥{R2_MIN_THRESHOLD} — using best "
                          f"available fallback (source="
                          f"{_best_fallback['metrics'].get('source')}, "
                          f"R²={_best_fallback['r2']:.3f}).")
                model     = _best_fallback["model"]
                sel_feats = _best_fallback["feats"]
                metrics   = _best_fallback["metrics"]
            else:
                print(f"ERROR: All model strategies exhausted for {loc} — cannot produce forecast.")
                return None, None, None

        # Record which model actually served this location's forecast.
        _persist_location_metrics(loc_clean, spec, s3_manager, metrics, silent=silent)

        # Predict
        if not silent:
            print(f"INFO: Predicting over {len(all_geos)} GEOS-CF rows…")

        model_features = _model_feature_names(model) or [_clean_name(f) for f in sel_feats]
        missing = [f for f in model_features if f not in all_geos.columns]
        if len(missing) == len(model_features):
            print(f"ERROR: No model features found in GEOS-CF data")
            return None, None, None
        if not silent:
            print(f"INFO: {len(model_features) - len(missing)}/{len(model_features)} "
                  f"features available for prediction"
                  + (f" (missing → NaN: {missing})" if missing else ""))

        target_mode = metrics.get("target", "absolute")
        preds = _predict_concentration(model, sel_feats, target_mode, all_geos, spec)
        all_geos["localised"] = preds

        n_valid = int(np.isfinite(preds).sum())
        if n_valid == 0:
            print(f"ERROR: No valid rows for prediction")
            return None, None, None
        if not silent:
            print(f"INFO: {n_valid}/{len(all_geos)} predictions "
                  f"({np.nanmin(preds):.2f} – {np.nanmax(preds):.2f}), target={target_mode}")

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

        print(f"INFO: Model performance for {loc} ({spec.upper()}): "
              f"{_format_metrics_summary(metrics)}")

        return result, metrics, model

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


# global model training

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
    
    # config

    FEATURES = [
        "no2", "o3", "co", "hcho", "no", "noy", "so2",
        "pm25_rh35", "pm25_rh35_gcc", "pm10_rh35",
        "pm25bc_rh35", "pm25du_rh35", "pm25nh4_rh35", "pm25nit_rh35",
        "pm25oc_rh35", "pm25soa_rh35", "pm25ss_rh35", "pm25su_rh35",
        "aod550_bc", "aod550_dust", "aod550_oc", "aod550_sala", "aod550_salc",
        "aod550_sulfate", "aod550_psc", "aod550_sla", "aod550_sna", "aod550_ss",
        "cldtt", "lev", "ps", "rh", "t10m", "tprec", "zpbl",
    ]
    
    # load data

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
    
    # temporal features

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
    
    # sampling (memory control)

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

    # no lag features here: they were sorted by hour-of-day not real time,
    # and serving has no recent obs to lag from, so they zero-filled and
    # biased every forecast toward 0
    lag_features = []

    gc.collect()
    
    # feature filtering

    temporal_features = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    available_features = [f for f in FEATURES + temporal_features + lag_features 
                          if f in df.columns]
    
    if verbose:
        print(f"\n✓ Total features: {len(available_features)}")
        print(f"  - Base: {len([f for f in FEATURES if f in df.columns])}")
        print(f"  - Temporal: {len(temporal_features)}")
        print(f"  - Lag/Rolling: {len(lag_features)}")
    
    # data cleaning

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
    
    # train/test split

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
    
    # model config

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
    
    # k-fold cross-validation

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
    
    # train final model

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
    
    # evaluation

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
    
    # save model

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


# enhanced localised forecast with global fallback

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
        # step 1: load observations
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
        
        # step 2: load geos-cf data
        if not silent:
            print(f"\n--- LOADING GEOS-CF DATA ---")
        
        # V1 (replay)
        geos_v1 = read_geos_cf(
            lon=lon, lat=lat, start=st, end=ed, version=1, verbose=not silent
        )
        
        # V2 (analysis + forecast)
        geos_v2 = read_geos_cf(
            lon=lon, lat=lat, start=st, end=datetime.now() + timedelta(days=5),
            version=2, verbose=not silent
        )
        
        # Combine V1 + V2
        frames = []
        if geos_v1 is not None and not geos_v1.empty:
            geos_v1["time"] = _normalize_time(geos_v1["time"])
            frames.append(geos_v1)
            if not silent:
                print(f"✓ V1: {len(geos_v1)} rows "
                      f"({geos_v1['time'].min()} → {geos_v1['time'].max()})")
        
        if geos_v2 is not None and not geos_v2.empty:
            geos_v2["time"] = _normalize_time(geos_v2["time"])
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

        # step 3: engineer features
        # full series, not the sparse subset, so lags mean the same at train and predict time
        if not silent:
            print(f"\n--- ENGINEERING FEATURES ---")

        geos_data_feat = _add_atmospheric_features(geos_data, spec=spec)

        if not silent:
            new_feat_count = len([c for c in geos_data_feat.columns if c not in geos_data.columns])
            print(f"✓ Added {new_feat_count} engineered features")
            print(f"  - Cyclical time (hour_sin/cos, doy_sin/cos)")
            print(f"  - Lag features (lag1h, lag3h, lag6h, lag24h)")
            print(f"  - Rolling statistics (roll3h, roll6h)")
            print(f"  - Atmospheric proxies (pbl_proxy, t2m_delta, etc.)")

        # step 4: merge for training
        if not silent:
            print(f"\n--- MERGING OBSERVATIONS & GEOS-CF ---")

        merged_train = (
            geos_data_feat.merge(obs_data[["time", "value"]], on="time", how="inner")
            .dropna(subset=["value"])
        )

        if len(merged_train) < 50:
            print(f"WARNING: Only {len(merged_train)} training samples (need ≥50)")
            print(f"  Using global model as fallback")
            merged_train = None
        else:
            if not silent:
                print(f"✓ Merged: {len(merged_train)} rows")
        
        # step 5: train local model
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

            n_splits = min(3, max(2, len(merged_train) // 300))
            kfold = TimeSeriesSplit(n_splits=n_splits)
            cv_r2s = []

            if not silent:
                print(f"\n--- TIME-SERIES CROSS-VALIDATION ({n_splits} folds) ---")

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
        
        # step 6: pick local or global
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
        
        # step 7: prepare data for prediction
        if not silent:
            print(f"\n--- PREPARING FOR PREDICTION ---")
        
        # reuse features from step 3 so lags match training time
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
        
        # step 8: generate predictions
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
        
        # step 9: interpolate, merge
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
        
        # step 10: summary
        perf_bits = [f"{k}={selected_metrics[k]}" for k in
                     ("source", "target_mode", "cv_r2", "test_r2", "test_rmse", "n_train")
                     if selected_metrics.get(k) is not None]
        print(f"INFO: Model performance for {loc} ({spec.upper()}): "
              f"{', '.join(perf_bits) if perf_bits else 'no metrics available'}")

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

