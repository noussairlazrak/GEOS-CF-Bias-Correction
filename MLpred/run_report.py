# -*- coding: utf-8 -*-
"""
run_report.py

"""

import os
import json
import functools
import subprocess
import datetime as dt
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


ENGINEERED_VARIABLES = {
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "weekday", "is_weekend",
    "lst_hour", "lst_weekday", "lst_is_weekend", "pbl_proxy", "wind_speed",
    "wind_dir_sin", "wind_dir_cos", "ventilation", "t2m_delta", "month", "hour",
}

METRIC_KEYS = ("CV_R2", "CV_RMSE", "CV_MAE", "R2", "RMSE", "MAE", "Train_R2",
               "n_train", "n_folds", "global_CV_R2", "global_CV_RMSE")
MODEL_KEYS = ("source", "approach", "algorithm", "target", "pipeline_version",
              "trained_at", "obs_start", "obs_end", "candidates")


def _safe(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except Exception as exc:
            print(f"WARNING: run report — {fn.__name__} failed: {exc}")
    return wrapper


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if not np.isfinite(v) else round(float(v), 4)
    if isinstance(v, float):
        return None if not np.isfinite(v) else round(v, 4)
    if isinstance(v, (pd.Timestamp, dt.datetime)):
        return str(v)
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    return v


def _feature_importance(model, features):
    """Normalised importance share per feature ({} when unavailable)."""
    members = getattr(model, "models", None) or [model]
    shares = []
    for m in members:
        imp = None
        booster = getattr(m, "booster_", None)
        if booster is not None:
            imp = booster.feature_importance(importance_type="gain")
        elif hasattr(m, "feature_importances_"):
            imp = m.feature_importances_
        if imp is None or len(imp) != len(features):
            continue
        imp = np.asarray(imp, dtype=float)
        if imp.sum() > 0:
            shares.append(imp / imp.sum())
    if not shares:
        return {}
    return {f: round(float(s), 4) for f, s in zip(features, np.mean(shares, axis=0))}


def _concentration_unit(spec):
    return "µg/m³" if spec == "pm25" else "ppb"


class RunReport:
    """Collects nodes and edges during a run; ``finalize`` writes the report."""

    def __init__(self, config=None, pipeline_version=None, run_id=None):
        self.started = dt.datetime.now()
        self.run_id = run_id or self.started.strftime("%Y%m%dT%H%M%S")
        self.config = config or {}
        self.pipeline_version = pipeline_version
        self.nodes = {}
        self.edges = []
        self.events = []
        self._model_ids = {}
        self._finalized = False
        self.run_node = f"run:{self.run_id}"
        self._node(self.run_node, "Run", started=str(self.started),
                   pipeline_version=pipeline_version, git_commit=self._git_commit(),
                   config=self.config)

    @staticmethod
    def _git_commit():
        try:
            repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            return subprocess.check_output(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                                           stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None

    def _node(self, nid, ntype, **props):
        node = self.nodes.setdefault(nid, {"id": nid, "type": ntype})
        node.update({k: _jsonable(v) for k, v in props.items() if v is not None})
        return nid

    def _edge(self, src, rel, dst, **props):
        edge = {"source": src, "relation": rel, "target": dst}
        edge.update({k: _jsonable(v) for k, v in props.items() if v is not None})
        if edge not in self.edges:
            self.edges.append(edge)

    @_safe
    def add_location(self, loc, lat=None, lon=None, obs_source=None):
        nid = self._node(f"location:{loc}", "Location", name=loc, lat=lat, lon=lon,
                         obs_source=obs_source)
        self._edge(self.run_node, "PROCESSED", nid)
        return nid

    @_safe
    def add_event(self, loc, status, reason=None, spec=None, obs_source=None):
        """Location-level outcome other than a forecast: skipped / failed."""
        self.events.append(dict(location=loc, spec=spec, status=status, reason=reason,
                                obs_source=obs_source, time=str(dt.datetime.now())))
        loc_id = self.add_location(loc, obs_source=obs_source)
        self._edge(self.run_node, status.upper(), loc_id, spec=spec, reason=reason)

    @_safe
    def add_observations(self, loc, spec, obs, obs_source=None, input_unit=None,
                         excluded=None):
        """Observation sample available to the models of *loc* / *spec*."""
        loc_id = self.add_location(loc, obs_source=obs_source)
        v = obs["value"].astype(float)
        nid = self._node(
            f"obs:{loc}:{spec}", "ObservationSet", location=loc, spec=spec,
            obs_source=obs_source, input_unit=input_unit, unit=_concentration_unit(spec),
            n_obs=int(v.notna().sum()), start=obs["time"].min(), end=obs["time"].max(),
            mean=v.mean(), p95=v.quantile(0.95), max=v.max(),
            excluded_periods=excluded or None,
        )
        self._edge(loc_id, "OBSERVED_BY", nid)
        return nid

    @_safe
    def add_model(self, loc, spec, model, features, metrics, role):
        """
        *role* is "trained" (fit this run) or "served" (made the forecast).
        A model both trained and served this run becomes one node.
        """
        metrics = metrics or {}
        key = (loc, spec, id(model))
        nid = self._model_ids.get(key)
        if nid is None:
            nid = f"model:{loc}:{spec}:{len(self._model_ids) + 1}"
            self._model_ids[key] = nid
        node = self.nodes.get(nid, {})
        roles = set(node.get("roles", [])) | {role}
        source = metrics.get("source")
        status = ("trained" if "trained" in roles or source == "trained" else "retained")
        self._node(
            nid, "Model", location=loc, spec=spec, status=status, roles=sorted(roles),
            served="served" in roles, n_features=len(features or []),
            **{k: metrics.get(k) for k in MODEL_KEYS},
            metrics={k: metrics.get(k) for k in METRIC_KEYS if metrics.get(k) is not None},
        )
        self._edge(f"location:{loc}", "HAS_MODEL", nid)
        if f"obs:{loc}:{spec}" in self.nodes:
            self._edge(nid, "TRAINED_ON", f"obs:{loc}:{spec}", n_samples=metrics.get("n_train"),
                       start=metrics.get("obs_start"), end=metrics.get("obs_end"))

        imp = _feature_importance(model, list(features or []))
        for rank, f in enumerate(sorted(features or [], key=lambda x: -imp.get(x, 0)), 1):
            vid = self._node(f"variable:{f}", "Variable", name=f,
                             kind="engineered" if f in ENGINEERED_VARIABLES else "geos_cf")
            self._edge(nid, "USES_VARIABLE", vid, importance=imp.get(f), rank=rank if imp else None)
        return nid

    @_safe
    def add_forecast(self, loc, spec, result, model, col="localised", obs_col="value"):
        """Forecast series *model* generated for *loc* / *spec*."""
        t = pd.to_datetime(result["time"])
        y = result[col].astype(float)
        has_obs = result[obs_col].notna() if obs_col in result.columns else pd.Series(False, index=result.index)
        now = pd.Timestamp(dt.datetime.now())
        future = t > now
        fid = self._node(
            f"forecast:{loc}:{spec}", "Forecast", location=loc, spec=spec,
            unit=_concentration_unit(spec), n_rows=len(result), n_valid=int(y.notna().sum()),
            n_with_obs=int(has_obs.sum()), n_forecast_only=int((~has_obs).sum()),
            n_future=int(future.sum()), start=t.min(), end=t.max(),
            future_mean=y[future].mean() if future.any() else None,
            future_max=y[future].max() if future.any() else None,
            mean=y.mean(), min=y.min(), max=y.max(),
        )
        self._edge(f"location:{loc}", "HAS_FORECAST", fid)
        mid = self._model_ids.get((loc, spec, id(model)))
        if mid:
            self._edge(mid, "GENERATED", fid)
        return fid

    @_safe
    def add_output(self, loc, species, path=None, s3_key=None, variant=None, n_rows=None,
                   sources=None):
        """A forecast file written for *loc*, holding *species*."""
        oid = self._node(f"file:{s3_key or path}", "OutputFile", location=loc, path=path,
                         s3_key=s3_key, variant=variant, n_rows=n_rows, species=list(species),
                         sources=sources)
        self.add_location(loc)
        for sp in species:
            fid = f"forecast:{loc}:{sp}"
            if fid not in self.nodes:
                self._node(fid, "Forecast", location=loc, spec=sp, sources=sources,
                           unit=_concentration_unit(sp))
                self._edge(f"location:{loc}", "HAS_FORECAST", fid)
            self._edge(fid, "WRITTEN_TO", oid)
        return oid

    def _by_type(self, ntype):
        return [n for n in self.nodes.values() if n["type"] == ntype]

    def summary(self):
        models = self._by_type("Model")
        status = Counter(e["status"] for e in self.events)
        return {
            "locations": len(self._by_type("Location")),
            "locations_with_forecast": len({n["location"] for n in self._by_type("Forecast")}),
            "skipped": status.get("skipped", 0),
            "failed": status.get("failed", 0),
            "models_trained": sum(m["status"] == "trained" for m in models),
            "models_retained": sum(m["status"] == "retained" for m in models),
            "models_served": sum(m.get("served", False) for m in models),
            "forecasts": len(self._by_type("Forecast")),
            "output_files": len(self._by_type("OutputFile")),
            "variables": len(self._by_type("Variable")),
            "observation_samples": sum(n.get("n_obs", 0) for n in self._by_type("ObservationSet")),
        }

    def to_graph(self):
        return {"run_id": self.run_id, "summary": self.summary(),
                "nodes": list(self.nodes.values()), "edges": self.edges,
                "events": self.events}

    def to_markdown(self):
        s = self.summary()
        run = self.nodes[self.run_node]
        f2 = lambda v: "–" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
        L = [f"# Forecast run {self.run_id}", "",
             f"- Started: {run.get('started')} · finished: {run.get('finished')} "
             f"({run.get('duration_min')} min)",
             f"- Pipeline version: {self.pipeline_version} · git commit: {run.get('git_commit')}",
             f"- Config: `{json.dumps(self.config, default=str)}`", "",
             "## Summary", "",
             "| Locations | With forecast | Skipped | Failed | Models trained | Models retained "
             "| Models served | Forecasts | Output files | Obs samples |",
             "|---|---|---|---|---|---|---|---|---|---|",
             "| " + " | ".join(str(s[k]) for k in (
                 "locations", "locations_with_forecast", "skipped", "failed", "models_trained",
                 "models_retained", "models_served", "forecasts", "output_files",
                 "observation_samples")) + " |", ""]

        rel = Counter((self.nodes[e["source"]]["type"], e["relation"], self.nodes[e["target"]]["type"])
                      for e in self.edges if e["source"] in self.nodes and e["target"] in self.nodes)
        L += ["## Knowledge graph", "", "```mermaid", "graph LR"]
        L += [f"  {a} -->|{r} ×{n}| {b}" for (a, r, b), n in sorted(rel.items())]
        L += ["```", ""]

        models = sorted(self._by_type("Model"), key=lambda m: (m["location"], m["spec"]))
        if models:
            L += ["## Models", "",
                  "| Location | Species | Status | Served | Source | Algorithm / target "
                  "| CV R² | CV RMSE | Train R² | Samples | Vars | Trained at | Obs period |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for m in models:
                mt = m.get("metrics", {})
                L.append("| " + " | ".join([
                    m["location"], m["spec"], m["status"], "yes" if m.get("served") else "no",
                    str(m.get("source", "–")), f"{m.get('algorithm', '–')} / {m.get('target', '–')}",
                    f2(mt.get("CV_R2", mt.get("R2"))), f2(mt.get("CV_RMSE", mt.get("RMSE"))),
                    f2(mt.get("Train_R2")), f2(mt.get("n_train")), str(m.get("n_features", "–")),
                    str(m.get("trained_at", "–")),
                    f"{str(m.get('obs_start', '–'))[:10]} → {str(m.get('obs_end', '–'))[:10]}",
                ]) + " |")
            L.append("")

        fcs = sorted(self._by_type("Forecast"), key=lambda f: (f["location"], f["spec"]))
        if fcs:
            gen = {e["target"]: e["source"] for e in self.edges if e["relation"] == "GENERATED"}
            files = defaultdict(list)
            for e in self.edges:
                if e["relation"] == "WRITTEN_TO":
                    files[e["source"]].append(self.nodes[e["target"]].get("variant") or "file")
            L += ["## Forecasts", "",
                  "| Location | Species | Unit | Rows | With obs | Future rows | Ends | "
                  "Future mean / max | Model | Files |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for f in fcs:
                L.append("| " + " | ".join([
                    f["location"], f["spec"], f.get("unit", "–"), f2(f.get("n_rows")),
                    f2(f.get("n_with_obs")), f2(f.get("n_future")), str(f.get("end", "–"))[:16],
                    f"{f2(f.get('future_mean'))} / {f2(f.get('future_max'))}",
                    (f"{self.nodes[gen[f['id']]].get('algorithm', '?')} "
                     f"({self.nodes[gen[f['id']]]['status']})") if f["id"] in gen else
                    ", ".join(f.get("sources", [])) or "–",
                    ", ".join(files.get(f["id"], [])) or "–",
                ]) + " |")
            L.append("")

        obs = sorted(self._by_type("ObservationSet"), key=lambda o: (o["location"], o["spec"]))
        if obs:
            L += ["## Observation samples", "",
                  "| Location | Species | Source | Samples | Period | Mean | P95 | Excluded |",
                  "|---|---|---|---|---|---|---|---|"]
            for o in obs:
                ex = "; ".join(f"{e['start']}→{e['end']} ({e.get('n', '?')})"
                               for e in o.get("excluded_periods", []) or []) or "–"
                L.append(f"| {o['location']} | {o['spec']} | {o.get('obs_source', '–')} | "
                         f"{o.get('n_obs')} | {str(o.get('start'))[:10]} → {str(o.get('end'))[:10]} | "
                         f"{f2(o.get('mean'))} {o.get('unit', '')} | {f2(o.get('p95'))} | {ex} |")
            L.append("")

        uses = defaultdict(list)
        for e in self.edges:
            if e["relation"] == "USES_VARIABLE":
                uses[e["target"]].append(e.get("importance"))
        if uses:
            L += ["## Variables", "",
                  "| Variable | Kind | Models using it | Mean importance share |",
                  "|---|---|---|---|"]
            mean_imp = lambda imps: float(np.mean([i for i in imps if i is not None] or [0]))
            for vid, imps in sorted(uses.items(), key=lambda kv: (-len(kv[1]), -mean_imp(kv[1]))):
                vals = [i for i in imps if i is not None]
                v = self.nodes[vid]
                L.append(f"| {v['name']} | {v.get('kind')} | {len(imps)} | "
                         f"{f2(float(np.mean(vals))) if vals else '–'} |")
            L.append("")

        if self.events:
            L += ["## Skipped / failed", "", "| Location | Species | Status | Reason |",
                  "|---|---|---|---|"]
            for e in self.events:
                reason = str(e.get("reason") or "–").replace("|", "/").replace("\n", " ")[:200]
                L.append(f"| {e['location']} | {e.get('spec') or '–'} | {e['status']} | {reason} |")
            L.append("")
        return "\n".join(L)

    def finalize(self, out_dir="reports", s3_manager=None, s3_prefix=None, clean_local=False):
        """Write knowledge_graph.json + report.md (and upload them). Runs once."""
        if self._finalized:
            return None
        self._finalized = True
        try:
            finished = dt.datetime.now()
            self._node(self.run_node, "Run", finished=str(finished),
                       duration_min=round((finished - self.started).total_seconds() / 60, 1),
                       summary=self.summary())
            run_dir = os.path.join(out_dir, f"run_{self.run_id}")
            os.makedirs(run_dir, exist_ok=True)
            paths = {"knowledge_graph.json": os.path.join(run_dir, "knowledge_graph.json"),
                     "report.md": os.path.join(run_dir, "report.md")}
            with open(paths["knowledge_graph.json"], "w") as f:
                json.dump(self.to_graph(), f, indent=1, default=str)
            with open(paths["report.md"], "w") as f:
                f.write(self.to_markdown())
            print(f"Run report written → {run_dir} ({self.summary()})")

            if s3_manager is not None and s3_prefix:
                for name, p in paths.items():
                    key = f"{s3_prefix}/run_{self.run_id}/{name}"
                    if s3_manager.upload_file(p, key):
                        print(f"Run report uploaded → {key}")
                if clean_local:
                    for p in paths.values():
                        os.remove(p)
            return paths
        except Exception as exc:
            print(f"WARNING: run report could not be written: {exc}")
            return None
