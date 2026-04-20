# -*- coding: utf-8 -*-
# ! /usr/bin/env python
""" 
mlpred.py

This file handles localized forecasts, based on GMAO's GEOS CF and OpenAQ data
.. codeauthor:: Christoph R Keller <christoph.a.keller@nasa.gov>
.. contributor:: Noussair Lazrak <noussair.lazrak@nyu.edu>
"""

# Standard library imports
import sys
import os
import re
import time
import pickle
import random
import io
from math import sqrt
from datetime import datetime, timedelta
import datetime as dt
import warnings
# Third-party imports
import numpy as np
import pandas as pd
import requests
import fsspec
import xarray as xr
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import shap
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import root_mean_squared_error, mean_squared_error, r2_score, accuracy_score, mean_absolute_error, median_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from dateutil.relativedelta import relativedelta
from pyod.models.iforest import IForest
from geopy.distance import geodesic
from bs4 import BeautifulSoup
from timezonefinder import TimezoneFinder
import pytz
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
from urllib.parse import urlencode
import urllib.request
import boto3
from botocore.exceptions import ClientError
import joblib 

# IPython and display
from IPython.display import HTML

# Local imports
sys.path.insert(1, 'MLpred')
from MLpred import mlpred
from MLpred import funcs
from MLpred.read_pandora import read_pandora, extract_metadata, convert_no2_mol_m3_to_ppbv

warnings.filterwarnings("ignore")

ZARR_TEMPLATE = [
    "geos-cf/zarr/geos-cf.met_tavg_1hr_g1440x721_x1.zarr",
    "geos-cf/zarr/geos-cf.chm_tavg_1hr_g1440x721_v1.zarr",
]
ZARR_TEMPLATE = ["geos-cf/zarr/geos-cf-rpl.zarr"]
S3_TEMPLATE = "s3://smce-geos-cf-public/geos-cf-rpl.zarr/"
S3_FORECASTS_TEMPLATE = "s3://smce-geos-cf-public/geos-cf-fcst-latest.zarr/"
S3_REPLAY_TEMPLATE = "s3://smce-geos-cf-public/geos-cf-ana-latest.zarr"
OPENDAP_TEMPLATE = "https://opendap.nccs.nasa.gov/dods/gmao/geos-cf/fcast/met_tavg_1hr_g1440x721_x1.latest"
M2_TEMPLATE = "/home/ftei-dsw/Projects/SurfNO2/data/M2/{c}/small/*.{c}.%Y%m*.nc4"
M2_COLLECTIONS = ["tavg1_2d_flx_Nx", "tavg1_2d_lfo_Nx", "tavg1_2d_slv_Nx"]
OPENAQ_TEMPLATE = "https://api.openaq.org/v2//measurements?date_from={Y1}-{M1}-01T00%3A00%3A00%2B00%3A00&date_to={Y2}-{M2}-01T00%3A00%3A00%2B00%3A00&limit=10000&page=1&offset=0&sort=asc&radius=1000&location_id={ID}&parameter={PARA}&order_by=datetime"

S3_V2_RPL = "s3://smce-geos-cf-public/geos-cf-v2-rpl.zarr"
S3_V2_COLS = "s3://smce-geos-cf-public/geos-cf-v2-rpl-cols.zarr"
S3_V2_FCST = "s3://smce-geos-cf-public/geos-cf-v2-fcst-latest.zarr/"


OPENAQAPI = "ae1be41f0d6e6400a0ad67ccdb6bea912c7787a14038038d94dfc1b2044f7cd4"
CACHE_DIR = "GEOS_CF"
MERRA2CNN = "https://aeronet.gsfc.nasa.gov/cgi-bin/web_print_air_quality_index"

DEFAULT_GASES = ["co", "hcho", "no", "no2", "noy", "o3"]

PPB2UGM3 = {"no2": 1.88, "o3": 1.97}
VVtoPPBV = 1.0e9


class ObsSiteList:
    def __init__(self, ifile=None):
        """
        Initialize ObsSiteList object (read from file if provided).
        """
        self._site_list = None
        if ifile is not None:
            self.load(ifile)

    def save(self, ofile="site_list.pkl"):
        """Write out a site_list, discarding all model and observation data beforehand (but keeping the trained XGBoost instances)
         `Generators` class: Contains medigan's public methods to facilitate users' automated sample generation workflows.
        Parameters
        ----------
        ofile: ofile
            Saves the list of sites to a pickle file
        """
        for isite in self._site_list:
            isite._obs = None
            isite._mod = None
        pickle.dump(self._site_list, open(ofile, "wb"), protocol=4)
        print("{} sites written to {}".format(len(self._site_list), ofile))
        return

    def load(self, ifile):
        """Reads a previously saved site list"""
        self._site_list = pickle.load(open(ifile, "rb"))
        print("Read {} sites from {}".format(len(self._site_list), ifile))
        return

    def filter_sites(self, year=2018, minobs=72, minvalue=15.0, silent=True):

        """Wrapper routine to get dataframe with average values for all sites with at least nobs observations for the first day of each month of the given year

        Parameters
        ----------
        year: year
            The year filter
        minobs: int
            The minimal observations to filter
        minvalue: float

        silent: bool
            Mute printed messages from the function

        Returns
        -------
        site_ids : pd.DataFrame
            A formatted observation data frame based on the filters
        """
        allmonths = []
        for imonth in tqdm(range(12)):
            testurl = "https://docs.openaq.org/v2/measurements?date_from={0:d}-{1:02d}-01T00%3A00%3A00%2B00%3A00&date_to={0:d}-{1:02d}-02T00%3A00%3A00%2B00%3A00&limit=100000&page=1&offset=0&sort=asc&parameter=no2&order_by=datetime".format(
                year, imonth + 1
            )
            allmonths.append(read_openaq(testurl, silent=silent))
        tmp = pd.concat(allmonths)
        cnt = tmp.groupby(["locationId", "unit"]).count().reset_index()
        sites = list(cnt.loc[cnt.value > minobs, "locationId"].values)
        subdf = tmp.loc[tmp["locationId"].isin(sites)].copy()
        meandf = subdf.groupby(["locationId", "unit"]).mean().reset_index()
        meandf.loc[meandf["unit"] == "µg/m³", "value"] = (
            meandf.loc[meandf["unit"] == "µg/m³", "value"] * 1.0 / 1.88
        )
        meandf.loc[meandf["unit"] == "ppm", "value"] = (
            meandf.loc[meandf["unit"] == "ppm", "value"] * 1000.0
        )
        site_ids = list(meandf.loc[meandf["value"] >= minvalue, "locationId"].values)
        print(
            "Found {} sites with average concentration above {} ppbv and more than {} observations".format(
                len(site_ids), minvalue, minobs
            )
        )
        self._minvalue = minvalue
        return site_ids

    def create_list(
        self,
        site_ids,
        minobs=240,
        silent=True,
        model_source="nc4",
        log=False,
        xgbparams={"booster": "gbtree", "eta": 0.5},
        **kwargs,
    ):
        """Create a list of observation sites by training all sites listed in site_ids that have at least minobs number of observations in the training window

        Parameters
        ----------
        site_ids: list
            list of OpenAQ site ids
        minobs: int
            The minimal observations to filter
        model_source: str
            Model sources
        silent: bool
            Mute printed messages from the function
        log: bool
            numpy.log for the training loop
        xgbparams: dict


        Returns
        -------
        site_ids : pd.DataFrame
            A formatted observation data frame based on the filters
        """
        self._site_list = []
        for i in tqdm(site_ids):
            isite = ObsSite(location_id=i, silent=silent, model_source=model_source)
            isite.read_obs(**kwargs)
            if isite._obs is None:
                if not isite._silent:
                    print("No observations found for site {}".format(i))
                continue
            if isite._obs.shape[0] < minobs:
                if not isite._silent:
                    print("Not enough observations found for site {}".format(i))
                continue
            # isite.read_mod()
            rc = isite.train(mindat=minobs, log=log, xgbparams=xgbparams)
            if rc == 0:
                self._site_list.append(isite)
        return

    def calc_ratios(self, start, end):

        """Get ratios between prediction and observation for each site in site_list'''


        Parameters
        ----------
        start: datetime
            The start date of training data set (GEOS-CF DATA and OpenAQ observation data)
        end: datetime
            The end date of training data set (GEOS-CF DATA and OpenAQ observation data)

        Returns
        -------
        dataframe
            a dataframe containing the ratios between prediction and observation for each site in site_list.
        """

        predictions = self.predict_sites(start, end)
        siteIds = []
        siteNames = []
        siteLats = []
        siteLons = []
        ratios = []
        meanObs = []
        meanPred = []
        for p in predictions:
            ip = predictions[p]
            idf = ip["prediction"]
            if idf is None:
                continue
            siteIds.append(p)
            siteNames.append(ip["name"])
            siteLats.append(ip["lat"])
            siteLons.append(ip["lon"])
            ratios.append(
                idf["observation"].values.mean() / idf["prediction"].values.mean()
            )
            meanObs.append(idf["observation"].values.mean())
            meanPred.append(idf["prediction"].values.mean())
        siteRatios = pd.DataFrame(
            {
                "Id": siteIds,
                "name": siteNames,
                "lat": siteLats,
                "lon": siteLons,
                "ratio": ratios,
                "obs": meanObs,
                "pred": meanPred,
            }
        )
        siteRatios["relChange"] = (siteRatios["ratio"] - 1.0) * 100.0
        return siteRatios

    def predict_sites(self, start, end):
        """Predict concentrations at all sites in the list of ObsSite objects

        Parameters
        ----------
        start: datetime
            The start date of training data set (GEOS-CF DATA and OpenAQ observation data)
        end: datetime
            The end date of training data set (GEOS-CF DATA and OpenAQ observation data)

        Returns
        -------
        list
            a list containing the prediction for each site
        """

        predictions = {}
        for isite in tqdm(self._site_list):
            isite.read_obs_and_mod(start=start, end=end)
            df = isite.predict(start=start, end=end)
            predictions[isite._id] = {
                "name": isite._name,
                "lat": isite._lat,
                "lon": isite._lon,
                "prediction": df,
            }
        return predictions

    def plot_deviation(
        self,
        siteRatios,
        title="NO2 deviation",
        minval=-30.0,
        maxval=30.0,
        mapbox_access_token=None,
    ):
        """Make global map showing deviation betweeen predictions and observations'''

        Parameters
        ----------
        siteRatios: list
            The site siteRatios betweeen predictions and observations for all sites
        title: str
            Map title
        minval: float

        maxval: float

        mapbox_access_token: str
            Mapbox token, Mapbox uses access tokens to associate API requests with your account. You can find your access tokens, create new ones, or delete existing ones on your Access Tokens page at mapbox.com


        Returns
        -------
        figure
            a map of deviation from all sites
        """

        siteRatios["text"] = [
            "{0:} (ID {1:}, Pred={2:.2f}ppbv, Deviation={3:.2f}%".format(i, j, k, l)
            for i, j, k, l in zip(
                siteRatios["name"],
                siteRatios["Id"],
                siteRatios["pred"],
                siteRatios["relChange"],
            )
        ]
        fig = go.Figure(
            data=go.Scattermapbox(
                lon=siteRatios["lon"],
                lat=siteRatios["lat"],
                text=siteRatios["text"],
                mode="markers",
                marker=go.scattermapbox.Marker(
                    size=siteRatios["pred"],
                    sizemode="area",
                    color=siteRatios["relChange"],
                    cmin=minval,
                    cmax=maxval,
                    colorscale="RdBu",
                    opacity=0.8,
                    autocolorscale=False,
                    reversescale=True,
                    colorbar_title=title,
                ),
                # name = siteRatios['name'],
            )
        )
        # fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(
            hovermode="closest",
            mapbox_accesstoken=mapbox_access_token,
            mapbox_style="dark",
        )
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        return fig


class ObsSite:
    def __init__(
        self,
        location_id,
        read_obs=False,
        silent=False,
        model_source="nc4",
        species="no2",
        **kwargs,
    ):
        """
        Initialize ObsSite object.
        """
        self._init_site(location_id, species, silent, model_source)
        if read_obs:
            self.read_obs(**kwargs)

    def read_obs_and_mod(self, **kwargs):
        """Convenience wrapper to read both observations and model data"""
        self.read_obs(**kwargs)
        self.read_mod(**kwargs)
        return

    def read_obs(self, data=None, resample=None, **kwargs):
        """Wrapper routine to read observations

        Parameters
        ----------
        data: dataframe
            check of the observation dataframe is not empty, otherwise this method will read observations from OpenAQ
        resample: str
            This provides the ability to resample observation to daily, n Days mean value, example: ("5D" means 5 days mean value resample)
        """
        source = kwargs.get("source")

        if data is None:
            data = pd.DataFrame()
            if source == "local":
                url = kwargs.get("url")
                time_col = kwargs.get("time_col")
                unit = kwargs.get("unit")
                date_format = kwargs.get("date_format")
                value_collum = kwargs.get("value_collum")
                lat_col = kwargs.get("lat_col")
                lon_col = kwargs.get("lon_col")
                species = kwargs.get("species")
                lname = kwargs.get("lname")
                lat = kwargs.get("lat")
                lon = kwargs.get("lon")
                remove_outlier = kwargs.get("remove_outlier")
                start = kwargs.get("start")
                end = kwargs.get("end")

                data = read_local_obs(
                    obs_url=url,
                    time_col=time_col,
                    date_format=date_format,
                    value_collum=value_collum,
                    lat_col=lat_col,
                    lon_col=lon_col,
                    species=species,
                    unit=unit,
                    lat=lat,
                    lon=lon,
                    start=start,
                    end=end,
                    remove_outlier = remove_outlier
                )

            elif source == "pandora":
                #print("pandora")
                url = kwargs.get("url")
                time_col = kwargs.get("time_col")
                date_format = kwargs.get("date_format")
                value_collum = kwargs.get("value_collum")
                lat_col = kwargs.get("lat_col")
                lon_col = kwargs.get("lon_col")
                species = kwargs.get("species")
                lname = kwargs.get("lname")

                data = read_pandora(
                    url=url, pollutant = species
                )


            else:
                data = pd.DataFrame()
                start_date = (
                    kwargs["start"] if "start" in kwargs else dt.datetime(2018, 1, 1)
                )
                end_date = kwargs["end"] if "end" in kwargs else dt.datetime(2024, 11, 11)

                month_difference = int((end_date.year - start_date.year) * 12 + end_date.month - start_date.month)


                three_month_periods = month_difference // 6

                for i in range(three_month_periods + 1):
                    period_start = start_date + relativedelta(months=6 * i)
                    
                    
                    period_end = min(start_date + relativedelta(months=6 * (i + 1)) - timedelta(days=1), end_date)
                    
                    print(f'getting openaq from {period_start} to {period_end}')
                    
                    
                    if not self._silent:
                        print(f"period retrieval {i + 1}: Start date - {period_start}, End date - {period_end}")

                    obs = self._read_openaq(start=period_start, end=period_end)
                    if obs is not None:
                        if data is None:
                            data = obs.copy()
                        else:
                            data = pd.concat([data, obs], ignore_index=True)


                folder_path = "obs/"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                data.to_csv(f"{folder_path}{self._id}.csv", index=False)

        if data is None:
            if not self._silent:
                print("Warning: no observations found!")
            return
        if "lat" not in data.columns:
            if not self._silent:
                print(
                    "Warning: no latitude entry found in observation data - cannot process information"
                )
            return
        if resample is not None:
            data = data.set_index("time").resample(resample).mean().reset_index()
            print("Resampled observation data to: {}".format(resample))
        ilat = np.round(data["lat"].median(), 2)
        ilon = np.round(data["lon"].median(), 2)
        iname = data["location"].values[0]
        if not self._silent:
            print(
                "Found {:d} observations for {:} (lon={:.2f}; lat={:.2f})".format(
                    data.shape[0], iname, ilon, ilat
                )
            )
        self._lat = ilat if self._lat is None else self._lat
        self._lon = ilon if self._lon is None else self._lon
        self._name = iname if self._name is None else self._name
        if iname != self._name and not self._silent:
            print(
                "Warning: new station name is {}, vs. previously {}".format(
                    iname, self._name
                )
            )
            self._name = iname
        self._obs = (
            self._obs.merge(data, how="outer") if self._obs is not None else data
        )
        return

    def read_mod(self, **kwargs):
        """Wrapper routine to read model data"""

        lon = kwargs.pop('lon', self._lon)
        lat = kwargs.pop('lat', self._lat)
        assert lon is not None and lat is not None

        if "start" not in kwargs:
            min_time = self._obs["time"].min()
            if min_time < dt.datetime(2018, 1, 1):
                kwargs["start"] = dt.datetime(2018, 1, 1)
            else:
                kwargs["start"] = min_time

        if "end" not in kwargs:
            kwargs["end"] = self._obs["time"].max()

        mod = self._read_model(lon, lat, **kwargs)

        self._mod = self._mod.merge(mod, how="outer") if self._mod is not None else mod

        return

    def train(
        self,
        target_var="value",
        skipvar=["time", "location", "lat", "lon"],
        mindat=None,
        test_size=0.3,
        log=False,
        inc=False,
        xgbparams={"booster": "gbtree"},
        model_type="xgboost-tuned",
        **kwargs,
    ):

        """Train XGBoost model using data in memory

        Parameters
        ----------
        target_var: str
            the target to be predicted
        skipvar: list
            list of values to be skipped in the training dataset
        mindat: float

        test_size: float
            the size of the testing data set (default value is 0.3 (30%))

        log: bool

        inc:bool

            set to false to predict concentration if target species is not a feature input

        xgbparams: list
            list of xgboost model parameters

        model_type:
            default: "xgboost-tuned" to select the tuned model or default model
        """

        dat = self._merge(**kwargs)

        if dat is None:
            return -2
        if mindat is not None:
            if dat.shape[0] < mindat:
                print(
                    "Warning: not enough data - only {} rows vs. {} requested".format(
                        dat.shape[0], mindat
                    )
                )
                return -1
        yvar = [target_var]
        blacklist = yvar + skipvar
        xvar = [i for i in dat.columns if i not in blacklist]
        X = dat[xvar]
        y = dat[yvar]
        fvar = None
        if inc:
            fvar = "pm25_rh35_gcc" if self._species == "pm25" else self._species
            if fvar not in X:
                print(
                    "Warning: target species is not an input feature - cannot do increment ML (set inc=False to predict concentration instead)"
                )
                return -1
            y = y.values.flatten() - X[fvar].values.flatten()
        if log:
            y = np.log(y)
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=test_size)

        if model_type == "Matrix":
            train = xgb.DMatrix(Xtrain, ytrain)
            if not self._silent:
                print("training model ...")
            bst = xgb.train(xgbparams, train)
            ptrain = bst.predict(xgb.DMatrix(Xtrain))
            ptest = bst.predict(xgb.DMatrix(Xtest))
            ytrainf = np.array(ytrain).flatten()
            ytestf = np.array(ytest).flatten()
            if log:
                ytrainf = np.exp(ytrainf)
                ytestf = np.exp(ytestf)
                ptrain = np.exp(ptrain)
                ptest = np.exp(ptest)
            if inc:
                ytrainf = ytrainf + np.array(Xtrain[fvar]).flatten()
                ytestf = ytestf + np.array(Xtest[fvar]).flatten()
                ptrain = ptrain + np.array(Xtrain[fvar]).flatten()
                ptest = ptest + np.array(Xtest[fvar]).flatten()
            if not self._silent:
                print("Training:")
                print("r2 = {:.2f}".format(r2_score(ytrainf, ptrain)))
                print("rmse = {:.2f}".format(mean_squared_error(ytrainf, ptrain)))
                print(
                    "nrmse = {:.2f}".format(
                        sqrt(mean_squared_error(ytrainf, ptrain)) / np.std(ytrainf)
                    )
                )
                print("nmb = {:.2f}".format(np.sum(ptrain - ytrainf) / np.sum(ytrainf)))
                print("Test:")
                print("r2 = {:.2f}".format(r2_score(ytestf, ptest)))
                print(
                    "nrmse = {:.2f}".format(
                        sqrt(mean_squared_error(ytestf, ptest)) / np.std(ytestf)
                    )
                )
                print("nmb = {:.2f}".format(np.sum(ptest - ytestf) / np.sum(ytestf)))

        if model_type == "xgboost-tuned":
            bst = xgb.XGBRegressor(
                colsample_bytree=0.3,
                learning_rate=0.01,
                max_depth=10,
                n_estimators=1000,
                verbosity=0,
            )
            prepared_model = bst.fit(Xtrain, ytrain)
            ypred = bst.predict(dat[X.columns])
            score = prepared_model.score(Xtest, ytest)
            target = prepared_model.predict(Xtest)

            if not self._silent:
                MSE = mean_squared_error(ytest, target)
                RMSE = root_mean_squared_error(ytest, target)
                MAE = mean_absolute_error(ytest, target)
                r2 = r2_score(ytest, target)

                # RMSE Computation
                print("Score: ", score)
                print("mean square error", MSE)
                print("Root mean square error", RMSE)
                print("MAE", MAE)
                print("R2", r2)

                print("Train Accuracy:", prepared_model.score(Xtrain, ytrain))
                print("Test Accuracy:", prepared_model.score(Xtest, ytest))

        self._bst = bst
        self._x = X
        self._xcolumns = X.columns
        self._log = log
        self._inc = inc
        self._fvar = fvar
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        self.ytrain = ytrain
        self.ytest = ytest
        return 0

    def predict(self, add_obs=True, model_type="xgboost-tuned", **kwargs):

        """Make prediction for given time window and return predicted values along with observations

        Parameters
        ----------
        add_obs: dataframe
            add observation data to geos-cf model data
        model_type: str
            predict using a predefined model, or default model, the predefined model is hyperparameter tuned
        minval: float



        Returns
        -------
        dataframe
            a dataframe containing the predictions vs observation
        """

        if add_obs:
            dat = self._merge(**kwargs)
        else:
            start = kwargs["start"] if "start" in kwargs else dat["time"].min()
            end = kwargs["end"] if "end" in kwargs else dat["time"].max()
            dat = self._mod.loc[
                (self._mod["time"] >= start) & (self._mod["time"] <= end)
            ].copy()
            if "value" not in dat:
                dat["value"] = [np.nan for i in range(dat.shape[0])]
        if dat is None:
            return None
        if model_type == "xgboost-tuned":
            pred = self._bst.predict(dat[self._xcolumns])
        if model_type == "Matrix":
            pred = self._bst.predict(xgb.DMatrix(dat[self._xcolumns]))

        if self._log:
            pred = np.exp(pred)
        if self._inc:
            pred = pred + dat[self._fvar]

        df = dat[["time", "value"]].copy()
        df["prediction"] = pred
        df.rename(columns={"value": "observation"}, inplace=True)
        return df

    def plot(
        self,
        df,
        y=["observation", "prediction"],
        ylabel=r"$\text{NO}_{2}\,[\text{ppbv}]$",
        **kwargs,
    ):

        """Make plot of prediction vs. observation, as generated by self.predict()

        Parameters
        ----------
        df: dataframe
            dataframe containing the prediction and observation values generated by the predict() method
        y: list
            list of y-axis,
        ylabel: str
            y-axis label

        Returns
        -------
        figure
            a timeseries figure of the predictions vs observation
        """

        title = "Site = {0} ({1:.2f}N, {2:.2f}E)".format(
            self._name, self._lat, self._lon
        )
        fig = px.line(
            df, x="time", y=y, labels={"value": ylabel}, title=title, **kwargs
        )
        fig.update_layout(xaxis_title="Date (UTC)", yaxis_title=ylabel)
        fig.update_layout(
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0
            ),
            legend_title="",
        )

        return fig

    def explain(self, plot=False, feature=False):
        """plot SHAP values to explain how the features are driving the predictions

        Parameters
        ----------
        plot: bool
            specify the type of the plot, examples: "waterfall", "beeswarm", "scatter"
        feature: str
            if the plot type is "scatter", please define the input/ feature you want to get analysis for.


        Returns
        -------
        figure
            a Shap figure of the predictions
        """

        if self._bst:
            explainer = shap.Explainer(self._bst)
            shap_values = explainer(self._x)
            if plot == "waterfall":
                shap.plots.waterfall(shap_values[0])
            if plot == "beeswarm":
                shap.plots.beeswarm(shap_values)
            if plot == "scatter":
                if feature:
                    try:
                        shap.plots.scatter(shap_values[:, feature], color=shap_values)
                    except:
                        print("feature not found!")

        else:
            print("Please train the model first")
        return

    def _merge(
        self,
        start=None,
        end=None,
        mod_blacklist=["lat", "lon", "lev"],
        interpolation=True,
    ):

        """Merge model and observation and limit to given time window

        Parameters
        ----------
        start: datetime
            The start date of training data set (GEOS-CF DATA and OpenAQ observation data)
        end: datetime
            The end date of training data set (GEOS-CF DATA and OpenAQ observation data)

        mod_blacklist: list
            list of blacklisted features

        Returns
        -------
        dataframe
            a dataframe containing a merged dataframe of the model and observations
        """

        if self._mod is None or self._obs is None:
            if not self._silent:
                print("Warning: cannot merge because mod or obs is None")
            return None
        
        # Toss model variables in blacklist
        ivars = [i for i in self._mod.columns if i not in mod_blacklist]
        
        # Merge all data
        mdat = (
            self._mod[ivars]
            .merge(self._obs, on=["time"], how="outer")
            .sort_values(by="time")
        )
        
        mdat = mdat.drop_duplicates(subset='time')
        mdat = mdat.sort_values(by='time')

        # Interpolate
        if interpolation:
            idat = mdat.set_index("time").interpolate(method="slinear").reset_index()
        else:
            idat = mdat

        # Resample to 1-hour intervals
        idat.set_index("time", inplace=True)
        idat_resampled = idat.resample('1H').mean(numeric_only=True).reset_index()

        # Filter by the provided start and end times
        if start is not None:
            idat_resampled = idat_resampled[idat_resampled['time'] >= pd.to_datetime(start)]
        
        if end is not None:
            idat_resampled = idat_resampled[idat_resampled['time'] <= pd.to_datetime(end)]

        if idat_resampled.shape[0] == 0:
            idat_resampled = None
        
        return idat_resampled


    def _init_site(self, location_id, species, silent, model_source):
        """Create an empty site object"""
        self._id = location_id
        self._species = species
        self._silent = silent
        self._modsrc = model_source
        self._lat = None
        self._lon = None
        self._name = None
        self._obs = None
        self._mod = None
        self._log = False
        self._inc = False
        self._fvar = None 
        return

    def _read_model(
            self,
            ilon,
            ilat,
            start,
            end,
            resample=None,
            source=None,
            template=None,
            collections=None,
            remove_outlier=0,
            gases=DEFAULT_GASES,
            **kwargs,
        ):
        """Read model data

        Parameters
        ----------
        ilon: float
            site longitude
        ilat:
            site latitude
        start: datetime
            The start date of training data set (GEOS-CF DATA)

        end: datetime
            The end date of training data set (GEOS-CF DATA)

        resample: str
            This provides the ability to resample observation to daily, n Days mean value, example: ("5D" means 5 days mean value resample)

        source: str
            specify the source file format (e.g. "opendap", "nc4", "zarr" for compressed format)

        template: str
            the link template for each model data file type

        collections: list
            model collection (e.g. "tavg1_2d_flx_Nx")

        gases: list
            list with gas names used to identify fields that need to be converted from v/v to ppbv

        Returns
        -------
        dataframe
            a dataframe containing the geos-cf model data
        """

        dfs = []
        source = self._modsrc if source is None else source
        
        
        ## Add xgc (aods, totcol), chem (species + noy), met
        if source == "opendap":
            template = OPENDAP_TEMPLATE if template is None else template
            template = template if isinstance(template, list) else [template]
            for t in template:
                if not self._silent:
                    print("Reading {}...".format(t))
                ids = (
                    xr.open_dataset(t)
                    .sel(lon=ilon, lat=ilat, lev=1, method="nearest")
                    .sel(time=slice(start, end))
                    .load()
                    .to_dataframe()
                    .reset_index()
                )
                dfs.append(ids)

        if source == "nc4":
            template = M2_TEMPLATE if template is None else template
            collections = M2_COLLECTIONS if collections is None else collections
            for c in collections:
                itemplate = template.replace("{c}", c)
                ifiles = start.strftime(itemplate)
                if not self._silent:
                    print("Reading {}...".format(c))
                ids = (
                    xr.open_mfdataset(ifiles)
                    .sel(lon=ilon, lat=ilat, method="nearest")
                    .sel(time=slice(start, end))
                    .load()
                    .to_dataframe()
                    .reset_index()
                )
                dfs.append(ids)


        if source in ["zarr", "s3"]:
            location_name = "loc_{:.2f}_{:.2f}".format(ilat, ilon).replace('.', '_').replace('-', 'm')
            
            # Determine storage source and parameters
            model_cache_source = kwargs.get('model_cache_source', 'local')
            s3_client = kwargs.get('s3_client', None)
            s3_bucket = kwargs.get('s3_bucket', None)
            s3_prefix = kwargs.get('s3_prefix', 'snwg_forecast_working_files/GEOS_CF/')
            
            # Setup local or S3 storage
            if model_cache_source == 's3':
                if not s3_client or not s3_bucket:
                    print("S3 cache requested but S3 client/bucket not provided. Falling back to local cache.")
                    model_cache_source = 'local'
            
            if model_cache_source == 'local':
                SAVED_FILES_DIR = "GEOS_CF"
                os.makedirs(SAVED_FILES_DIR, exist_ok=True)
            
            base_start = datetime(2018, 1, 1)
            base_end = (datetime.today() + timedelta(days=5)).date()
            location_name = f"loc_{ilat}_{ilon}".replace('.', '_').replace('-', 'm') 
            
            if model_cache_source == 'local':
                csv_path = os.path.join(SAVED_FILES_DIR, f"{location_name}.csv")
            else:
                csv_path = None  


            force_full_download = True

            # load existing cache
            if model_cache_source == 'local':
                df_existing = None
                if os.path.exists(csv_path):
                    print(f"Found existing CSV for {location_name}. Checking time coverage...")
                    df_existing = pd.read_csv(csv_path, parse_dates=["time"])
            else:  # S3
                if funcs.check_s3_file_status(s3_client, s3_bucket, s3_prefix, location_name):
                    print(f"Found existing S3 CSV for {location_name}. Checking time coverage...")
                    df_existing = funcs.read_s3_file(s3_client, s3_bucket, s3_prefix, location_name)
                else:
                    df_existing = None

            if df_existing is not None:
                t_min = df_existing["time"].min().date()
                t_max = df_existing["time"].max().date()

                if t_min <= base_start.date():
                    print("CSV has historical data. Will only fetch forecast data.")
                    df_all = [df_existing]

                    try:
                        ipath_fc = fsspec.get_mapper(S3_FORECASTS_TEMPLATE)
                        ds_fc = xr.open_zarr(ipath_fc)

                        df_fc = (ds_fc.sel(lon=ilon, lat=ilat, lev=1, method='nearest')
                                        .load()
                                        .to_dataframe()
                                        .reset_index())

                        df_fc["time"] = pd.to_datetime(df_fc["time"])
                        df_all.append(df_fc)

                    except Exception as e:
                        print(f"Error reading forecast data: {e}")

                    df_combined = pd.concat(df_all, ignore_index=True)
                    df_combined = df_combined.drop_duplicates(subset="time").sort_values("time")
                    print(f'df_combined: {df_combined["time"].max()}')

                    # Save to cache (local or S3)
                    if model_cache_source == 'local':
                        df_combined.to_csv(csv_path, index=False)
                        print(f"Updated CSV saved: {csv_path}")
                    else:  # S3
                        funcs.upload_to_s3(csv_path, s3_client, s3_bucket, s3_key)

                    df_filtered = df_combined[(df_combined["time"] >= start) & (df_combined["time"] <= end)]
                    dfs.append(df_combined)

                else:
                    print("CSV does not contain full history. Will fetch full data.")
                    force_full_download = True

            else:
                print("No CSV found. Will fetch full data.")
                force_full_download = True

            if force_full_download:
                df_all = []
                sources = {
                    "historical": S3_TEMPLATE,
                    "replay": S3_REPLAY_TEMPLATE,
                    "forecast": S3_FORECASTS_TEMPLATE,
                }

                for label, template in sources.items():
                    print(f"Reading {label} data from {template}...")
                    try:
                        ipath = fsspec.get_mapper(template)
                        ds = xr.open_zarr(ipath)

                        df = (ds.sel(lon=ilon, lat=ilat, lev=1, method='nearest')
                                .load()
                                .to_dataframe()
                                .reset_index())

                        df["time"] = pd.to_datetime(df["time"])
                        df_all.append(df)

                    except Exception as e:
                        print(f"Error reading {label}: {e}")

                if df_all:
                    df_combined = pd.concat(df_all, ignore_index=True)
                    df_combined = df_combined.drop_duplicates(subset="time").sort_values("time")

                    # Save to cache (local or S3)
                    if model_cache_source == 'local':
                        df_combined.to_csv(csv_path, index=False)
                        print(f"Full dataset saved to {csv_path}")
                    else:  # S3
                        funcs.upload_to_s3(csv_path, s3_client, s3_bucket, s3_key)

                    df_filtered = df_combined[(df_combined["time"] >= start) & (df_combined["time"] <= end)]
                    dfs.append(df_filtered)
                else:
                    print("No data could be fetched.")

        if source == "local":
            url = kwargs.get("url")
            if not self._silent:
                print(f"Reading csv file: {url}")
            df = pd.read_csv(url)
            df["time"] = pd.to_datetime(df["time"])
            ids = df[(df["time"] >= start) & (df["time"] <= end)]
            if not ids.empty:
                dfs.append(ids)

        if source == "pandora":
            url = kwargs.get("url")
            if not self._silent:
                print(f"Reading csv file: {url}")
            df = pd.read_csv(url)
            df["time"] = pd.to_datetime(df["date"])
            df = df.drop(columns=['date'])
            ids = df[(df["time"] >= start) & (df["time"] <= end)]
            if not ids.empty:
                dfs.append(ids)

        mod = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        mod["time"] = pd.to_datetime(mod["time"])
        mod["month"] = mod["time"].dt.month
        mod["hour"] = mod["time"].dt.hour
        mod["weekday"] = mod["time"].dt.weekday

        if resample is not None:
            mod = mod.set_index("time").resample(resample).mean().reset_index()
            print("Resampled model data to: {}".format(resample))

        for g in gases:
            if g in mod:
                if not self._silent:
                    print("Convert from v/v to ppbv: {}".format(g))
                mod[g] = mod[g] * VVtoPPBV

        """
        mod['t10m'] = mod['t10m'].combine_first(mod['t'])
        mod['u10m'] = mod['u10m'].combine_first(mod['u'])
        mod['v10m'] = mod['v10m'].combine_first(mod['v'])
        
        mod['aod550_sala'] = mod['aod550_sna'].combine_first(mod['aod550_sna'])
        mod['aod550_salc'] = mod['aod550_ss'].combine_first(mod['aod550_ss'])
        mod = mod[['time', 'aod550_bc', 'aod550_dust', 'aod550_oc', 'aod550_sala',
       'aod550_salc', 'cldtt', 'co', 'hcho', 'lat', 'lev',
       'lon', 'no', 'no2', 'noy', 'o3', 'pm25_rh35_gcc', 'ps', 'rh', 't10m',
       'tprec',
       'tropcol_so2', 'u10m', 'v10m', 'zpbl']]
        
        print(mod.columns)
        """
        print(mod["time"].max())
        
        return mod


    def _read_openaq(
        self=None, start=None, end=None, normalize=False, **kwargs
    ):
        """Read OpenAQ observations and convert to ppbv

        Parameters
        ----------
        self: object, optional
            The instance of the class (if used as a method)
        start: datetime
            The start date of training data set (GEOS-CF DATA)
        end: datetime
            The end date of training data set (GEOS-CF DATA)
        normalize: bool
            if True, normalize the observations values with standard deviation

        Returns
        -------
        dataframe
            a dataframe containing the observation data
        """

        # If self is None, remove all self and replace with args parameters
        if self is None:
            id_ = kwargs.get("_id", None)
            species = kwargs.get("_species", None)
            silent = kwargs.get("_silent", None)
        else:
            id_ = self._id
            species = self._species
            silent = self._silent

        end = start + relativedelta(years=1) if end is None else end
        url = (
            OPENAQ_TEMPLATE.replace("{ID}", str(id_))
            .replace("{PARA}", species)
            .replace("{Y1}", str(start.year))
            .replace("{M1}", "{:02d}".format(start.month))
            .replace("{D1}", "{:02d}".format(start.day))
            .replace("{Y2}", str(end.year))
            .replace("{M2}", "{:02d}".format(end.month))
            .replace("{D2}", "{:02d}".format(end.day))
        )
        allobs = read_openaq(id_,start,end,parameter=species,silent=silent,remove_outlier=0,api_key=OPENAQAPI,chunk_days=360, **kwargs)

        if allobs is None:
            return None

        obs = allobs.loc[
            (allobs["parameter"] == species)
            & (~np.isnan(allobs["value"]))
            & (allobs["value"] >= 0.0)
        ].copy()

        # convert everything to ppbv
        if species != "pm25":
            assert species in PPB2UGM3

            conv_factor = PPB2UGM3[species]
            if not self._silent:
                print(f" converting to ppbv with conv_factor {conv_factor}")
            obs.loc[obs["unit"] == "ppm", "value"] = (
                obs.loc[obs["unit"] == "ppm", "value"] * 1000.0
            )
            obs.loc[obs["unit"] == "µg/m³", "value"] = (
                obs.loc[obs["unit"] == "µg/m³", "value"] * 1.0 / conv_factor
            )

        # subset to relevant columns
        outobs = obs[["time", "location", "value"]].copy()

        if normalize:
            outobs["value"] = (outobs["value"] - outobs["value"].mean()) / outobs[
                "value"
            ].std()

        if (
            "coordinates.latitude" in obs.columns
            and "coordinates.longitude" in obs.columns
        ):
            outobs["lat"] = obs["coordinates.latitude"]
            outobs["lon"] = obs["coordinates.longitude"]
        else:
            if not silent:
                print("Warning: no coordinates in dataset")

        return outobs

    def explain_model(model, X, plot, feature=False):
        """explain model via Shap values

        Parameters
        ----------
        model: model
            predefined model in memory

        X: dataframe
            dataframe in memory

        plot: str
            type of plot to be returned (e.g. "waterfall", "beeswarm", "scatter")

        feature:str
            when using scatter plot, please specify the feature to run shap analysis for

        Returns
        -------
        figure
            a Shap values plot
        """
        try:
            explainer = shap.Explainer(model)
            shap_values = explainer(X)
            if plot == "waterfall":
                shap.waterfall_plot(shap_values[0])
            if plot == "beeswarm":
                shap.beeswarm_plot(shap_values)
            if plot == "scatter":
                if feature:
                    shap.plots.scatter(shap_values[:, feature], color=shap_values)

        except:
            print("Warning: Model Error")

    def save_model(model, model_data=False, save_data=False, name=False):
        if name is False:
            name = "pretrained_model"
        pickle.dump(model, open(name + ".sav", "wb"))
        print("Model saved")
        if save_data:
            model_data.to_csv("model_data.csv")
            print("Model data saved")
        return

    def load_model(name=False, **kwargs):
        loaded_model = pickle.load(open(name, "rb"))
        return loaded_model

    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    def gridSerch(self, model, X_train, Y_train, **kwargs):
        print("Tunning the model hyper parameter for this location")
        params = {
            "max_depth": [3, 5, 6, 10, 15, 20],
            "learning_rate": [0.01, 0.1, 0.2, 0.3],
            "subsample": np.arange(0.5, 1.0, 0.1),
            "colsample_bytree": np.arange(0.4, 1.0, 0.1),
            "colsample_bylevel": np.arange(0.4, 1.0, 0.1),
            "n_estimators": [100, 500, 1000],
        }
        clf = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            scoring="neg_mean_squared_error",
            n_iter=25,
            verbose=1,
        )
        clf.fit(X_train, Y_train)
        print("Best parameters:", clf.best_params_)
        print("Lowest RMSE: ", (-clf.best_score_) ** (1 / 2.0))
        return clf

    def plot_intervals(
        self, predictions, mid=False, start=None, stop=None, title=None, **kwargs
    ):
        predictions = (
            predictions.loc[start:stop].copy()
            if start is not None or stop is not None
            else predictions.copy()
        )
        data = []

        """Lower Trace"""

        trace_low = go.Scatter(
            x=predictions.index,
            y=predictions["lower"],
            fill="tonexty",
            line=dict(color="darkblue"),
            fillcolor="rgba(173, 216, 230, 0.4)",
            showlegend=True,
            name="lower",
        )
        """Upper Trace"""
        trace_high = go.Scatter(
            x=predictions.index,
            y=predictions["upper"],
            fill=None,
            line=dict(color="orange"),
            showlegend=True,
            name="upper",
        )

        data.append(trace_high)
        data.append(trace_low)

        if mid:
            trace_mid = go.Scatter(
                x=predictions.index,
                y=predictions["mid"],
                fill=None,
                line=dict(color="green"),
                showlegend=True,
                name="mid",
            )
            data.append(trace_mid)

        """Actual Values Trace"""
        trace_actual = go.Scatter(
            x=predictions.index,
            y=predictions["mid"],
            fill=None,
            line=dict(color="black"),
            showlegend=True,
            name="middle",
        )
        data.append(trace_actual)

        """Observation Values Trace"""
        trace_actual = go.Scatter(
            x=predictions.index,
            y=predictions["value"],
            fill=None,
            line=dict(color="red"),
            showlegend=True,
            name="observation",
        )
        data.append(trace_actual)

        """prediction Values Trace"""
        bias_corrected = go.Scatter(
            x=predictions.index,
            y=predictions["bias_corrected"],
            fill=None,
            line=dict(color="blue"),
            showlegend=True,
            name="prediction",
        )
        data.append(bias_corrected)

        """Title and customization"""
        layout = go.Layout(
            height=900,
            width=1400,
            title=dict(text="Prediction Intervals" if title is None else title),
            yaxis=dict(title=dict(text="NO2 ppvb")),
            xaxis=dict(
                rangeselector=dict(
                    buttons=list(
                        [
                            dict(count=1, label="1d", step="day", stepmode="backward"),
                            dict(count=7, label="1w", step="day", stepmode="backward"),
                            dict(
                                count=1, label="1m", step="month", stepmode="backward"
                            ),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all"),
                        ]
                    )
                ),
                rangeslider=dict(visible=True),
                type="date",
            ),
        )

        fig = go.Figure(data=data, layout=layout)

        fig["layout"]["font"] = dict(size=20)
        fig.layout.template = "plotly_white"
        return fig

    def ConfidenceIntervals(
        self,
        LOWER_ALPHA=0.15,
        UPPER_ALPHA=0.85,
        N_ESTIMATORS=1000,
        MAX_DEPTH=5,
        LEARNING_RATE=0.01,
        colsample_bytree=0.3,
        OUTPUT="plot",
        **kwargs,
    ):
        """explain model via Shap values

        Parameters
        ----------
        model: model
            predefined model in memory

        X: dataframe
            dataframe in memory

        plot: str
            type of plot to be returned (e.g. "waterfall", "beeswarm", "scatter")

        feature:str
            when using scatter plot, please specify the feature to run shap analysis for

        Returns
        -------
        figure
            a Shap values plot
        """

        lower_model = GradientBoostingRegressor(
            loss="quantile",
            alpha=LOWER_ALPHA,
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
        )

        mid_model = xgb.XGBRegressor(
            loss="ls",
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            colsample_bytree=colsample_bytree,
            learning_rate=LEARNING_RATE,
            verbosity=0,
        )

        upper_model = GradientBoostingRegressor(
            loss="quantile",
            alpha=UPPER_ALPHA,
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
        )

        lower_model.fit(self.Xtrain, self.ytrain)
        mid_model.fit(self.Xtrain, self.ytrain)
        upper_model.fit(self.Xtrain, self.ytrain)

        predictions = pd.DataFrame(self.ytest)
        predictions["lower"] = lower_model.predict(self.Xtest)
        predictions["mid"] = mid_model.predict(self.Xtest)
        predictions["upper"] = upper_model.predict(self.Xtest)
        predictions["bias_corrected"] = self._bst.predict(self.Xtest)

        dat = self._merge()
        df = dat[["time", "value"]].copy()
        to_plot = df.merge(predictions)
        to_plot["timestamp"] = pd.to_datetime(to_plot["time"], unit="s")
        to_plot = to_plot.set_index(pd.DatetimeIndex(to_plot["timestamp"]))
        to_plot.dropna()
        to_plot.sort_index(inplace=True)

        if OUTPUT == "PLOT":
            to_plot.sort_index().dropna().resample("2D").mean()
            fig = self.plot_intervals(to_plot, start="2021-01-19", stop="2022-02-28")
            return fig
        if OUTPUT == "dataframe":
            return to_plot


def get_localised_forecast( loc='', spec='no2', lat=0.0, lon=0.0, mod_src='s3', obs_src='pandora', openaq_id=None, GEOS_CF=None, OBS_URL=None, st=None, ed=None, resamp='1h', unit='ppb', interpol=True, rmv_out=True, time_col='time', date_fmt='%Y-%m-%d %H:%M', obs_val_col='unit', lat_col=None, lon_col=None, silent=False, force_retrain=True, **kwargs ):
    """
    Generate localized air quality forecasts using GEOS-CF model data and observations.
    
    Training Strategy:
    ------------------
    1. Train base model on GEOS-CF V1 historical data (2018 to recent)
    2. Apply transfer learning using GEOS-CF V2 recent data (last 60 days)
    3. Generate forecasts using GEOS-CF V2 forecast data only
    
    This approach leverages the longer historical record from V1 while adapting
    to the improved V2 model for current and future predictions.
    
    Parameters
    ----------
    loc : str
        Location name
    spec : str
        Species to forecast (e.g., 'no2', 'o3', 'pm25')
    lat, lon : float
        Location coordinates
    mod_src : str
        Model data source (default: 's3')
    obs_src : str
        Observation data source (default: 'pandora')
    st, ed : datetime
        Start and end dates for data
    force_retrain : bool
        If True, retrain model even if saved model exists
    
    Returns
    -------
    merged_data : pd.DataFrame
        Combined observations, model data, and forecasts
    metrics : dict
        Model performance metrics (RMSE, R2, MAE)
    model : LGBMRegressor
        Trained model object
    """
   

    # Set defaults if not provided
    if st is None:
        st = dt.datetime(2018, 1, 1)
    if ed is None:
        ed = dt.datetime.today()
    if GEOS_CF is None:
        GEOS_CF = ''
    if OBS_URL is None:
        OBS_URL = ''

    # Data preparation
    try:
        obs_dt = pd.DataFrame()
        site = mlpred.ObsSite(openaq_id, model_source=mod_src, species=spec, observation_source=obs_src)
        site._silent = silent
        
        # Read observations
        site.read_obs(source=obs_src, url=OBS_URL, time_col=time_col, date_format=date_fmt, 
                      value_collum=obs_val_col, lat_col=lat_col, lon_col=lon_col, species=spec, 
                      lat=lat, lon=lon, unit=unit, remove_outlier=rmv_out, **kwargs)
        obs_dt = site._obs
        
        if not silent:
            print(f"Reading GEOS-CF data (V1 historical + V2 recent/forecast)...")
        
        # Define transition point between V1 and V2
        v1_end = datetime.now() - timedelta(days=60)  # V1 ends 60 days ago
        v2_start = v1_end  # V2 starts where V1 ends (no gap, no overlap)
        
        # Fetch V1 historical data (start to 60 days ago)
        if not silent:
            print(f"Fetching V1 data: {st.strftime('%Y-%m-%d')} to {v1_end.strftime('%Y-%m-%d')}")
        df_v1 = mlpred.read_geos_cf(lon=lon, lat=lat, start=st, end=v1_end, version=1)
        
        # Fetch V2 recent + forecast data (60 days ago to future)
        if not silent:
            print(f"Fetching V2 data: {v2_start.strftime('%Y-%m-%d')} to present + 5 days forecast")
        df_v2 = mlpred.read_geos_cf(lon=lon, lat=lat, start=v2_start, end=None, version=2)
        
        if not silent:
            print(f"V1 data: {len(df_v1)} rows ({df_v1['time'].min()} to {df_v1['time'].max()})")
            print(f"V2 data: {len(df_v2)} rows ({df_v2['time'].min()} to {df_v2['time'].max()})")
        
        # Combine 
        site._mod = pd.concat([df_v1, df_v2], ignore_index=True).drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)
        
        # Extract 
        now = datetime.now()
        pred_dt = site._mod[site._mod['time'] >= now - timedelta(hours=1)].copy()
        
        if not silent:
            print(f"Combined continuous dataframe: {len(site._mod)} rows ({site._mod['time'].min()} to {site._mod['time'].max()})")
            print(f"Forecast portion: {len(pred_dt)} rows (from {pred_dt['time'].min()} to {pred_dt['time'].max()})")
            print(f"V2 forecast data: {len(pred_dt)} rows from {pred_dt['time'].min()} to {pred_dt['time'].max()}")
        
        # Merge observations with training data
        merged = site._merge(interpolation=interpol)
        loc_dt = merged.dropna(subset=["value"])
        
        pred_dt["time"] = pd.to_datetime(pred_dt["time"])
        if not silent:
            print(f'pred_dt max time: {pred_dt["time"].max()}')
            
    except Exception as e:
        print(f"Error. Data preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

    yvar = "value"
    fvar = "pm25_rh35_gcc" if getattr(site, '_species', 'no2') == "pm25" else getattr(site, '_species', 'no2').lower()
    if mod_src == "pandora": fvar = "pandora"

    try:
        diff = loc_dt[fvar].mean() / loc_dt[yvar].mean()
        funcs.log_if_condition((diff > 2), f"UNIT Error. GEOS CF IS HIGHER BY: {diff} IN LOCATION: {loc} SPECIES: {spec.lower()}")
    except Exception:
        pass

    skipvar = ["time", "location", "lat", "lon"]
    blacklist = skipvar + [yvar]

    try:
        if rmv_out:
            conc_obs = loc_dt[yvar].values.reshape(-1, 1)
            nan_mask = ~np.isnan(conc_obs).flatten()
            conc_obs = conc_obs[nan_mask]
            loc_dt = loc_dt[nan_mask]
            if len(conc_obs) == 0: raise ValueError("No valid observations remaining after NaN removal")
            model_IF = IForest(contamination=0.05)
            model_IF.fit(conc_obs)
            anomalies = model_IF.predict(conc_obs)
            loc_dt = loc_dt[anomalies != 1]
    except Exception as e:
        print(f"Error. Outlier removal failed: {e}")
        return None, None, None

    # Feature selection
    try:
        corrs = loc_dt.corr()[yvar].drop(yvar)
        sel_feats = [col for col in corrs.index if abs(corrs[col]) > 0.1 and col not in blacklist]
        x = loc_dt[sel_feats]
        y = loc_dt[yvar]
    except Exception as e:
        print(f"Error. Feature selection failed: {e}")
        return None, None, None

    # Split and clean
    try:
        tx, vx, ty, vy = train_test_split(x, y, test_size=0.3, random_state=7)
        tx, ty = funcs.clean_data(tx, ty)
        vx, vy = funcs.clean_data(vx, vy)
        tx = funcs.clean_feature_names(tx)
        vx = funcs.clean_feature_names(vx)
    except Exception as e:
        print(f"Error Data split/clean failed: {e}")
        return None, None, None

    model_v1_path = f"MODELS/lgbm_{loc}_{spec}_v1.joblib"
    model_v2_path = f"MODELS/lgbm_{loc}_{spec}_v2.joblib"
    feature_v1_file = f"MODELS/lgbm_{loc}_{spec}_v1_features.pkl"
    feature_file = f"MODELS/lgbm_{loc}_{spec}_features.pkl"
    
    # skip re-training
    use_pretrained_v2 = os.path.exists(model_v2_path) and not force_retrain
    
    if use_pretrained_v2:
        if not silent:
            print(f"Loading pretrained V2 model from {model_v2_path}")
        model_lgb = joblib.load(model_v2_path)
        try:
            import pickle
            if os.path.exists(feature_file):
                sel_feats = pickle.load(open(feature_file, 'rb'))
                if not silent:
                    print(f"Loaded feature set: {len(sel_feats)} features")
            else:
                # Fall back to current feature selection
                corrs = loc_dt.corr()[yvar].drop(yvar)
                sel_feats = [col for col in corrs.index if abs(corrs[col]) > 0.1 and col not in blacklist]
        except Exception as e:
            if not silent:
                print(f"WARNING: Could not load features: {e}, using current feature selection")
            corrs = loc_dt.corr()[yvar].drop(yvar)
            sel_feats = [col for col in corrs.index if abs(corrs[col]) > 0.1 and col not in blacklist]
    else:
        # ============================================================
        # Train or Load V1 Base Model
        # ============================================================
        use_pretrained_v1 = os.path.exists(model_v1_path) and not force_retrain
        
        if use_pretrained_v1:
            if not silent:
                print(f"Loading pretrained V1 model from {model_v1_path}")
            model_lgb_v1 = joblib.load(model_v1_path)
            
            # Load the feature set that was used for V1
            try:
                import pickle
                if os.path.exists(feature_v1_file):
                    sel_feats_v1 = pickle.load(open(feature_v1_file, 'rb'))
                    if not silent:
                        print(f"Loaded V1 feature set: {len(sel_feats_v1)} features")
                else:
                    # Fall back to current feature selection
                    if not silent:
                        print(f"WARNING: V1 feature file not found, using current feature selection")
                    corrs_v1 = loc_dt.corr()[yvar].drop(yvar)
                    sel_feats_v1 = [col for col in corrs_v1.index if abs(corrs_v1[col]) > 0.1 and col not in blacklist]
            except Exception as e:
                if not silent:
                    print(f"WARNING: Could not load V1 features: {e}")
                corrs_v1 = loc_dt.corr()[yvar].drop(yvar)
                sel_feats_v1 = [col for col in corrs_v1.index if abs(corrs_v1[col]) > 0.1 and col not in blacklist]
        else:
            # Train new V1 model
            if not silent:
                print(f"Training base model on V1 data")
            
            # Filter training data for V1 period (use v1_end from data fetching)
            v1_end_date = datetime.now() - timedelta(days=60)  # Same as v1_end from data fetching
            v1_train_mask = loc_dt['time'] <= v1_end_date
            loc_dt_v1 = loc_dt[v1_train_mask].copy()
            
            if len(loc_dt_v1) < 100:
                if not silent:
                    print(f"WARNING: Insufficient V1 training data ({len(loc_dt_v1)} samples). Using all available data.")
                loc_dt_v1 = loc_dt.copy()
            
            # Feature selection on V1 data
            corrs_v1 = loc_dt_v1.corr()[yvar].drop(yvar)
            sel_feats_v1 = [col for col in corrs_v1.index if abs(corrs_v1[col]) > 0.1 and col not in blacklist]
            x_v1 = loc_dt_v1[sel_feats_v1]
            y_v1 = loc_dt_v1[yvar]
            
            tx_v1, vx_v1, ty_v1, vy_v1 = train_test_split(x_v1, y_v1, test_size=0.3, random_state=7)
            tx_v1, ty_v1 = funcs.clean_data(tx_v1, ty_v1)
            vx_v1, vy_v1 = funcs.clean_data(vx_v1, vy_v1)
            tx_v1 = funcs.clean_feature_names(tx_v1)
            vx_v1 = funcs.clean_feature_names(vx_v1)
            
            rs_params = {
                'num_leaves': [15, 31, 63],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [100, 300, 500],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0, 0.1, 0.5]
            }
            
            try:
                rs = RandomizedSearchCV(lgb.LGBMRegressor(verbosity=-1), rs_params, n_iter=10, cv=3, 
                                       scoring='neg_mean_squared_error', n_jobs=-1, error_score='raise', random_state=42)
                rs.fit(tx_v1, ty_v1)
                if not silent:
                    print(f"Best V1 features: {rs.best_params_}")
                
                from lightgbm import early_stopping
                model_lgb_v1 = lgb.LGBMRegressor(**rs.best_params_, verbosity=-1)
                model_lgb_v1.fit(tx_v1, ty_v1, eval_set=[(vx_v1, vy_v1)], callbacks=[early_stopping(stopping_rounds=20, verbose=0)])
                
                # Save V1 model and features for future 
                joblib.dump(model_lgb_v1, model_v1_path)
                import pickle
                pickle.dump(sel_feats_v1, open(feature_v1_file, 'wb'))
                
                if not silent:
                    print(f"V1 base model saved to {model_v1_path}")
                    print(f"V1 features saved to {feature_v1_file}")
                    print(f"Next time, V1 training will be skipped")
            except Exception as e:
                print(f"Error !V1 model training failed: {e}")
                import traceback
                traceback.print_exc()
                return None, None, None
        
        # Step 2: Transfer learning on V2 data
        if not silent:
            print(f"Step 2: Transfer learning with V2 data...")
        
        # Filter for V2 period (recent data - same as v2_start from data fetching)
        v2_start_date = datetime.now() - timedelta(days=60)  # Same as v2_start from data fetching
        v2_mask = loc_dt['time'] >= v2_start_date
        loc_dt_v2 = loc_dt[v2_mask].copy()
        
        if len(loc_dt_v2) < 50:
            if not silent:
                print(f"WARNING: Limited V2 data ({len(loc_dt_v2)} samples). Using V1 model only.")
            model_lgb = model_lgb_v1
        else:
            # Use same features as GEOS-CF V1
            sel_feats = [f for f in sel_feats_v1 if f in loc_dt_v2.columns]
            x_v2 = loc_dt_v2[sel_feats]
            y_v2 = loc_dt_v2[yvar]
            
            tx_v2, vx_v2, ty_v2, vy_v2 = train_test_split(x_v2, y_v2, test_size=0.3, random_state=7)
            tx_v2, ty_v2 = funcs.clean_data(tx_v2, ty_v2)
            vx_v2, vy_v2 = funcs.clean_data(vx_v2, vy_v2)
            tx_v2 = funcs.clean_feature_names(tx_v2)
            vx_v2 = funcs.clean_feature_names(vx_v2)
            
            # Fine-tune the V1 model with V2 data (lower learning rate for transfer learning)
            model_lgb = lgb.LGBMRegressor(
                **{**rs.best_params_, 'learning_rate': rs.best_params_.get('learning_rate', 0.05) * 0.5},
                verbosity=-1
            )
            
            # Initialize with V1 model weights by using init_model parameter
            model_lgb.fit(tx_v2, ty_v2, eval_set=[(vx_v2, vy_v2)], 
                         init_model=model_lgb_v1,
                         callbacks=[early_stopping(stopping_rounds=10, verbose=0)])
            
            # Save V2 model and features
            joblib.dump(model_lgb, model_v2_path)
            import pickle
            pickle.dump(sel_feats, open(feature_file, 'wb'))
            
            if not silent:
                print(f"V2 transfer learning model saved to {model_v2_path}")
                print(f"V2 features saved to {feature_file}")
                print(f"Next time, both V1 and V2 training will be skipped (instant load!)")
        
        # Use the combined feature set for consistency
        sel_feats = sel_feats_v1

    # Validate final model performance
    try:
        # Use most recent data for validation
        recent_mask = loc_dt['time'] >= (datetime.now() - timedelta(days=90))
        loc_dt_recent = loc_dt[recent_mask].copy() if recent_mask.sum() > 50 else loc_dt.copy()
        
        x_recent = loc_dt_recent[sel_feats]
        y_recent = loc_dt_recent[yvar]
        
        _, vx_final, _, vy_final = train_test_split(x_recent, y_recent, test_size=0.3, random_state=7)
        vx_final, vy_final = funcs.clean_data(vx_final, vy_final)
        vx_final = funcs.clean_feature_names(vx_final)
        
        vy_pred = model_lgb.predict(vx_final)
        rmse = round(root_mean_squared_error(vy_final, vy_pred), 2)
        r2 = round(r2_score(vy_final, vy_pred), 2)
        mae = round(mean_absolute_error(vy_final, vy_pred), 2)
        metrics = {'RMSE': rmse, 'R2': r2, 'MAE': mae}
        if not silent:
            print(f"Final model performance: RMSE={rmse}, R2={r2}, MAE={mae}")
            print(f"Selected features: {sel_feats}")
        funcs.log_if_condition((r2 < 0.5), f"MODEL Error !MODEL RUNS POORLY IN THIS LOCATION: R2: {r2} ; RMSE: {rmse} IN LOCATION: {loc} SPECIES: {spec.lower()}")
    except Exception as e:
        print(f"Error. Metrics calculation failed: {e}")
        metrics = None

    # Make predictions on V2 forecast data only
    try:
        pred_dt["time"] = [dt.datetime(i.year, i.month, i.day, i.hour, 0, 0) for i in pred_dt["time"]]
        
        # Ensure selected features exist in pred_dt
        missing_feats = [f for f in sel_feats if f not in pred_dt.columns]
        if missing_feats:
            if not silent:
                print(f"WARNING: Missing features in forecast data: {missing_feats}")
            # Only use features that exist in pred_dt
            sel_feats = [f for f in sel_feats if f in pred_dt.columns]
            if not silent:
                print(f"Reduced to {len(sel_feats)} features that exist in forecast data")
        
        # Check for non-null values in each feature
        null_counts = pred_dt[sel_feats].isnull().sum()
        
        # Filter out features that are usualy not available
        features_with_data = [f for f in sel_feats if null_counts[f] < len(pred_dt)]
        completely_null_feats = [f for f in sel_feats if null_counts[f] == len(pred_dt)]
        
        if completely_null_feats:
            if not silent:
                print(f"WARNING: Removing {len(completely_null_feats)} features with NO forecast data: {completely_null_feats}")
            sel_feats = features_with_data
            if not silent:
                print(f"[NFO: Using {len(sel_feats)} features with available forecast data")
        
        # Verify we have at least some features
        if len(sel_feats) == 0:
            raise ValueError(f"No features available for prediction! All features are missing from forecast data.")
        
        if not silent:
            remaining_null_counts = pred_dt[sel_feats].isnull().sum()
            if (remaining_null_counts > 0).any():
                print(f": Null counts in remaining features:\n{remaining_null_counts[remaining_null_counts > 0]}")
            print(f": pred_dt shape: {pred_dt.shape}, columns: {len(pred_dt.columns)}")
            print(f": Final features for prediction ({len(sel_feats)}): {sel_feats}")
        
        # Warn if using significantly fewer features than trained on
        original_feature_count = len([f for f in sel_feats if f not in completely_null_feats]) if 'completely_null_feats' in locals() else len(sel_feats)
        if completely_null_feats and not silent:
            pct_removed = (len(completely_null_feats) / (len(sel_feats) + len(completely_null_feats))) * 100
            print(f"WARNING: Predicting with {pct_removed:.1f}% fewer features than trained model expects")
            print(f"Model may be less accurate. Consider retraining with only V2-available features.")
        
        # Create mask for rows with all non-null features
        pred_mask = pred_dt[sel_feats].notnull().all(axis=1)
        n_valid = pred_mask.sum()
        
        if not silent:
            print(f"Valid prediction rows (all features non-null): {n_valid} / {len(pred_dt)}")
        
        pred_dt["localised"] = np.nan
        
        if n_valid > 0:
            # Get the subset to predict on
            X_pred = pred_dt.loc[pred_mask, sel_feats].copy()
            
            if not silent:
                print(f" X_pred shape before clean_feature_names: {X_pred.shape}")
                print(f" X_pred columns: {list(X_pred.columns)}")
                print(f" X_pred dtypes:\n{X_pred.dtypes}")
                print(f" X_pred sample:\n{X_pred.head()}")
            
            # Verify X_pred is valid
            if X_pred.empty or X_pred.shape[0] == 0 or X_pred.shape[1] == 0:
                if not silent:
                    print(f"Error. X_pred is empty after masking! Shape: {X_pred.shape}")
                    print(f": pred_mask details: {pred_mask.value_counts()}")
                raise ValueError(f"Prediction data is empty after filtering. Shape: {X_pred.shape}")
            
            X_pred = funcs.clean_feature_names(X_pred)
            
            if not silent:
                print(f":X_pred shape after clean_feature_names: {X_pred.shape}")
                print(f":X_pred columns after cleaning: {list(X_pred.columns)}")
            
            # Verify X_pred is still valid after cleaning
            if X_pred.empty or X_pred.shape[0] == 0 or X_pred.shape[1] == 0:
                if not silent:
                    print(f"Error. X_pred became empty after clean_feature_names! Shape: {X_pred.shape}")
                raise ValueError(f"Prediction data became empty after cleaning feature names. Shape: {X_pred.shape}")
            
            # LightGBM models
            model_features = model_lgb.feature_name_
            X_pred_aligned = X_pred.copy()
            
            missing_from_pred = [f for f in model_features if f not in X_pred_aligned.columns]
            if missing_from_pred:
                if not silent:
                    print(f"WARNING: Adding {len(missing_from_pred)} missing features as zeros: {missing_from_pred[:5]}...")
                for feat in missing_from_pred:
                    X_pred_aligned[feat] = 0.0
            
            # Reorder columns to match model's expected feature order
            X_pred_aligned = X_pred_aligned[model_features]
            
            if not silent:
                print(f": Aligned prediction data shape: {X_pred_aligned.shape}")
            
            # predictions loop
            predictions = model_lgb.predict(X_pred_aligned)
            pred_dt.loc[pred_mask, "localised"] = predictions
            
            if not silent:
                print(f"Generated {len(predictions)} localised forecasts from V2 data")
        else:
            if not silent:
                print(f"WARNING: No valid rows for prediction - all forecast rows have missing features")
                print(f"Forecast data time range: {pred_dt['time'].min()} to {pred_dt['time'].max()}")
                print(f"Consider checking if forecast data is available for future dates")
        
        # Add raw model forecast columns to pred_dt for merging
        for col in ["no2", "o3", "pm25_rh35_gcc", "pm25_rh35", "rh", "t", "tprec", "hcho", "co"]:
            if col not in pred_dt.columns:
                pred_dt[col] = np.nan
    except Exception as e:
        print(f"Error. Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None, metrics, model_lgb

    # ================================================================
    # Generate localised predictions for the entire continuous dataframe
    # (historical + recent + forecast - all from site._mod)
    # ================================================================
    if not silent:
        print(f"Generating localised predictions for entire time series...")
    
    try:
        # Use the continuous dataframe (V1 + V2) from site._mod
        continuous_dt = site._mod.copy()
        continuous_dt["time"] = pd.to_datetime(continuous_dt["time"])
        continuous_dt["time"] = [dt.datetime(i.year, i.month, i.day, i.hour, 0, 0) for i in continuous_dt["time"]]
        
        # Filter to features available in continuous data
        cont_sel_feats = [f for f in sel_feats if f in continuous_dt.columns]
        
        # Check for completely null features
        cont_null_counts = continuous_dt[cont_sel_feats].isnull().sum()
        cont_features_with_data = [f for f in cont_sel_feats if cont_null_counts[f] < len(continuous_dt)]
        cont_completely_null = [f for f in cont_sel_feats if cont_null_counts[f] == len(continuous_dt)]
        
        if cont_completely_null and not silent:
            print(f"Removing {len(cont_completely_null)} features with no data: {cont_completely_null}")
        
        cont_sel_feats = cont_features_with_data
        
        if len(cont_sel_feats) > 0:
            # Create mask for valid rows
            cont_pred_mask = continuous_dt[cont_sel_feats].notnull().all(axis=1)
            n_cont_valid = cont_pred_mask.sum()
            
            if not silent:
                print(f"Valid prediction rows: {n_cont_valid} / {len(continuous_dt)}")
            
            continuous_dt["localised"] = np.nan
            
            if n_cont_valid > 0:
                # Prepare data for prediction
                X_cont = continuous_dt.loc[cont_pred_mask, cont_sel_feats].copy()
                X_cont = funcs.clean_feature_names(X_cont)
                
                # Align features with model expectations
                model_features = model_lgb.feature_name_
                X_cont_aligned = X_cont.copy()
                
                missing_from_cont = [f for f in model_features if f not in X_cont_aligned.columns]
                if missing_from_cont:
                    for feat in missing_from_cont:
                        X_cont_aligned[feat] = 0.0
                
                X_cont_aligned = X_cont_aligned[model_features]
                
                # Make predictions for entire time series
                cont_predictions = model_lgb.predict(X_cont_aligned)
                continuous_dt.loc[cont_pred_mask, "localised"] = cont_predictions
                
                if not silent:
                    print(f"Generated {len(cont_predictions)} localised predictions for continuous time series")
                    print(f"Time range: {continuous_dt['time'].min()} to {continuous_dt['time'].max()}")
        else:
            if not silent:
                print(f"WARNING: No valid features for predictions")
            continuous_dt["localised"] = np.nan
    except Exception as e:
        if not silent:
            print(f"WARNING: Prediction generation failed: {e}")
        continuous_dt = site._mod.copy()
        continuous_dt["time"] = pd.to_datetime(continuous_dt["time"])
        continuous_dt["localised"] = np.nan

    obs = site._obs[["time", "value"]].copy()
    obs["time"] = [dt.datetime(i.year, i.month, i.day, i.hour, 0, 0) for i in obs["time"]]

    def clean_merged_values(df):
        if {'value_x', 'value_y'}.issubset(df.columns):
            df['value'] = df['value_x'].combine_first(df['value_y'])
            return df.drop(['value_x', 'value_y'], axis=1)
        return df

    # Merge continuous GEOS-CF data (V1+V2) with observations
    try:
        # Standardize time columns
        continuous_dt["time"] = pd.to_datetime(continuous_dt["time"]).dt.floor("H")
        obs["time"] = pd.to_datetime(obs["time"]).dt.floor("H")
        
        # Merge the single continuous dataframe with observations
        merged_data = funcs.merge_dataframes([continuous_dt, obs], index_col="time", resample=resamp, how="outer")
        merged_data = clean_merged_values(merged_data)
        
        # Interpolate observation values to fill nighttime gaps (for Pandora)
        if interpol and obs_src.lower() in ['pandora', 'sun-dependent']:
            if 'value' in merged_data.columns:
                n_missing_before = merged_data['value'].isna().sum()
                if n_missing_before > 0:
                    # Set time as index temporarily for time-aware interpolation
                    if 'time' in merged_data.columns:
                        merged_data_indexed = merged_data.set_index('time')
                        merged_data_indexed['value'] = merged_data_indexed['value'].interpolate(
                            method='time', 
                            limit=12,  # Maximum 12 consecutive hours to interpolate
                            limit_direction='both'
                        )
                        merged_data = merged_data_indexed.reset_index()
                    else:
                        # Fallback to linear interpolation if time is not a column
                        merged_data['value'] = merged_data['value'].interpolate(
                            method='linear', 
                            limit=12,
                            limit_direction='both'
                        )
                    
                    n_missing_after = merged_data['value'].isna().sum()
                    n_filled = n_missing_before - n_missing_after
                    if not silent and n_filled > 0:
                        print(f"Interpolated {n_filled} observation values to fill nighttime gaps (limit: 12 hours)")
        
        # Ensure all forecast columns are present in merged_data
        for col in ["no2", "localised", "o3", "pm25_rh35_gcc", "rh", "t10m", "tprec", "hcho", "value"]:
            if col not in merged_data.columns:
                merged_data[col] = np.nan
        if not silent:
            print("merged_data time range:", merged_data["time"].min(), merged_data["time"].max())


    except Exception as e:
        print(f"Error. Merging failed: {e}")
        import traceback
        traceback.print_exc()
        merged_data = None
    return merged_data, metrics, model_lgb


## General Functions
def read_openaq(sensor_id,start,end,parameter='o3',silent=False,remove_outlier=0,api_key=OPENAQAPI,chunk_days=30,**kwargs):

    base_url = f"https://api.openaq.org/v3/sensors/{sensor_id}/measurements"
    
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    
    all_data = []
    
    while start <= end:
        chunk_end = min(start + pd.Timedelta(days=chunk_days), end)
        
        params = {
            "datetime_from": start.strftime("%Y-%m-%d"),
            "datetime_to": chunk_end.strftime("%Y-%m-%d"),
            "limit": 1000,
            "order_by": "datetime",
            "sort": "desc"
        }
        
        url = f"{base_url}?{urlencode(params)}"

        if not silent:
            print(f"Querying {url}")
        
        retries = 0
        max_retries = 4

        while retries < max_retries:
            headers = {'X-API-Key': api_key}
            r = requests.get(url, headers=headers)
            
            if r.status_code == 200:
                results = r.json().get("results", [])
                if results:
                    all_data.extend(results)
                    
                    next_url = r.json().get("meta", {}).get("next")
                    if next_url:
                        url = next_url
                    else:
                        break
                else:
                    if not silent:
                        print(f"No results found for date range {start} to {chunk_end}")
                    break
            elif r.status_code == 429:
                retries += 1
                if retries < max_retries:
                    time.sleep(5)
            else:
                if not silent:
                    print(f"Error pulling data from OpenAQ: Status code {r.status_code}")
                break

        start = chunk_end + pd.Timedelta(days=1)

    if not all_data:
        if not silent:
            print("Warning: no OpenAQ data found for specified parameters")
        return None

    try:
        allobs = pd.json_normalize(all_data)
        
        allobs = allobs.loc[
            (allobs["value"] >= 0.0) & (~np.isnan(allobs["value"]))
        ].copy()
        
        allobs["time"] = pd.to_datetime(allobs["period.datetimeFrom.utc"], format="%Y-%m-%dT%H:%M:%SZ")
        allobs["unit"] = allobs["parameter.units"]
        
        
        if remove_outlier > 0:
            std = allobs["value"].std()
            mn = allobs["value"].mean()
            minobs = mn - remove_outlier * std
            maxobs = mn + remove_outlier * std
            norig = allobs.shape[0]
            allobs = allobs.loc[
                (allobs["value"] >= minobs) & (allobs["value"] <= maxobs)
            ].copy()
            if not silent:
                nremoved = norig - allobs.shape[0]
                print(
                    f"Removed {nremoved:.0f} of {norig:.0f} values because considered outliers ({float(nremoved) / float(norig) * 100.0:.2f}%)"

                )
        
        allobs["location"] = "openaq"
        allobs = allobs[["time", "value", "unit", "parameter.name", "period.datetimeFrom.local","location"]]
        allobs.columns = ["time", "value", "unit", "parameter", "local_time","location"]
        
        return allobs

    except Exception as e:
        if not silent:
            print(f"Warning: An error occurred while processing the data - {str(e)}")
        return None

def read_local_obs( obs_url=None, time_col="Time", date_format="%m/%d/%Y %H:%M:%S", value_collum= None, lat_col=None, lon_col=None, species=None, silent=True, start = None, end = None, remove_outlier=False, rename_column=None, unit=None, lat=None, lon=None, **kwargs, ):

    col_name = rename_column if rename_column else "value"

    allobs = pd.read_csv(obs_url)

    allobs = allobs.loc[
        (allobs[value_collum] >= 0.0) & (~np.isnan(allobs[value_collum]))
    ].copy()

    allobs["time"] = [dt.datetime.strptime(i, date_format) for i in allobs[time_col]]

    # allobs[col_name] = allobs[value_collum]

    conversion_unit = "ppbv" if species != "pm25" else "ugm3"

    allobs[col_name] = convert_pollutant(
        species, allobs[value_collum], unit, conversion_unit
    )

    allobs["lat"] = allobs[lat_col] if lat_col else lat
    allobs["lon"] = allobs[lon_col] if lon_col else lon
    location_name = obs_url.split("/")[-1].split("_")[0]
    allobs["location"] = location_name
    allobs = allobs[["time", col_name, "lat", "lon", "location"]]

    if remove_outlier:
        if not silent:   
            print("removing outlier ....")
        std = allobs[col_name].std()
        mn = allobs[col_name].mean()
        minobs = mn - remove_outlier * std
        maxobs = mn + remove_outlier * std
        norig = allobs.shape[0]

        allobs = allobs.loc[
            (allobs[col_name] >= minobs) & (allobs[col_name] <= maxobs)
        ].copy()

 
        z_scores = np.abs(stats.zscore(allobs[col_name]))
        threshold = 3
        outlier_indices = np.where(z_scores > threshold)[0]  
        allobs = allobs.drop(allobs.index[outlier_indices]) 
        
        if start is not None:
            start = pd.to_datetime(start)
        if end is not None:
            end = pd.to_datetime(end)

        if start is not None and end is not None:
            allobs = allobs[(allobs['time'] >= start) & (allobs['time'] <= end)]

    return allobs

# read_pandora, extract_metadata, and convert_no2_mol_m3_to_ppbv have been
# moved to MLpred/read_pandora.py and are imported at the top of this file.


def read_geos_fp_cnn(base_url=MERRA2CNN, site=None, frequency=30, lat=None, lon=None, silent=True, skip_geosfp = False):
    
    end_date = datetime.today() + timedelta(days=5)
    start_date = end_date - timedelta(days=frequency)
    all_data = pd.DataFrame()

    # Fetch MERRA2 data for the date range
    if not skip_geosfp:
        for n in range(frequency + 3):
            date = start_date + timedelta(days=n)
            url = f"{base_url}?year={date.year}&month={date.month}&day={date.day}&site={site}"

            if not silent:
                print(f"Fetching data for {date.strftime('%Y-%m-%d')} from {url}")

            try:
                response = requests.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                text = soup.body.get_text()

                df = pd.read_csv(io.StringIO(text))
                if df.empty:
                    if not silent:
                        print("Warning: Empty DataFrame retrieved.")
                    continue

                reshaped_data = []
                for _, row in df.iterrows():
                    try:
                        base_date = pd.to_datetime(row['UTC_DATE'])
                        for hour_offset, conc_col, aqi_col in zip(
                            [1, 4, 7, 10, 13, 16, 19, 22],
                            ['3HR_PM_CONC_CNN(130)', '3HR_PM_CONC_CNN(430)', '3HR_PM_CONC_CNN(730)', 
                             '3HR_PM_CONC_CNN(1030)', '3HR_PM_CONC_CNN(1330)', '3HR_PM_CONC_CNN(1630)', 
                             '3HR_PM_CONC_CNN(1930)', '3HR_PM_CONC_CNN(2230)'],
                            ['3HR_AQI(130)', '3HR_AQI(430)', '3HR_AQI(730)', '3HR_AQI(1030)', 
                             '3HR_AQI(1330)', '3HR_AQI(1630)', '3HR_AQI(1930)', '3HR_AQI(2230)']
                        ):
                            timestamp = base_date + timedelta(hours=hour_offset)
                            reshaped_data.append({
                                'time': timestamp,
                                'pm25_conc_cnn': row.get(conc_col, None),
                                'pm25_aqi': row.get(aqi_col, None),
                                'Station': row.get('Station', None),
                                'Site_Name': row.get('Site_Name', None)
                            })
                    except Exception as row_err:
                        if not silent:
                            print(f"Row processing Error. {row_err}")
                        continue

                reshaped_df = pd.DataFrame(reshaped_data)
                all_data = pd.concat([all_data, reshaped_df], ignore_index=True)

            except Exception as e:
                if not silent:
                    print(f"Failed to fetch data for {date.strftime('%Y-%m-%d')}: {e}")
                    continue

    # If all_data is empty, fill with nulls for expected columns so it can merge with geos_cf
    if all_data.empty:
        if not silent:
            print("No MERRA2 data retrieved. Filling with nulls for merge.")
        # Create a DataFrame with expected columns and nulls
        all_data = pd.DataFrame({
            'time': pd.date_range(start=start_date, end=end_date, freq='3H'),
            'pm25_conc_cnn': [np.nan] * len(pd.date_range(start=start_date, end=end_date, freq='3H')),
            'pm25_aqi': [np.nan] * len(pd.date_range(start=start_date, end=end_date, freq='3H')),
            'Station': [None] * len(pd.date_range(start=start_date, end=end_date, freq='3H')),
            'Site_Name': [None] * len(pd.date_range(start=start_date, end=end_date, freq='3H'))
        })


    if not silent:
        print("Requesting GEOS-CF...")

    geos_cf = mlpred.read_geos_cf(
        lon=lon,
        lat=lat,
        start=datetime.today() - timedelta(days=frequency),
        end=datetime.today() + timedelta(days=10),
        version = 2
    )

    # Merge and process data
    merg = funcs.merge_dataframes([all_data, geos_cf], "time", resample="3h", how="outer")
    print(merg.columns.to_list())

    # Fill missing MERRA-2 values with GEOS-CF values after merge
    if not silent:
        print("Filling missing MERRA-2 values with GEOS-CF data...")
    
    # Track what was filled for reporting
    filled_counts = {}
    
    # Fill pm25_conc_cnn with GEOS-CF PM2.5 if missing
    if 'pm25_conc_cnn' in merg.columns and 'pm25_rh35' in merg.columns:
        missing_mask = merg['pm25_conc_cnn'].isna()
        n_missing = missing_mask.sum()
        if n_missing > 0:
            merg.loc[missing_mask, 'pm25_conc_cnn'] = merg.loc[missing_mask, 'pm25_rh35']
            filled_counts['pm25_conc_cnn'] = n_missing
            if not silent:
                print(f"  - Filled {n_missing} missing pm25_conc_cnn values with pm25_rh35_gcc")

    # Calculate NowCast and AQI
    species_map = {'PM2.5': 'pm25_rh35', 'NO2': 'no2', 'O3': 'o3'}
    avg_hours = {'NO2': 3, 'O3': 1}
    merg = funcs.calculate_nowcast(merg, species_columns=species_map, avg_hours=avg_hours)

    # Fill missing pm25_aqi with GEOS-CF PM25_NowCast_AQI if available
    if 'pm25_aqi' in merg.columns and 'PM25_NowCast_AQI' in merg.columns:
        missing_mask = merg['pm25_aqi'].isna()
        n_missing = missing_mask.sum()
        if n_missing > 0:
            merg.loc[missing_mask, 'pm25_aqi'] = merg.loc[missing_mask, 'PM25_NowCast_AQI']
            filled_counts['pm25_aqi'] = n_missing
            if not silent:
                print(f"  - Filled {n_missing} missing pm25_aqi values with PM25_NowCast_AQI")
    
    # Summary of filling operations
    if not silent and filled_counts:
        print(f"Summary: Filled missing values - {filled_counts}")
    elif not silent:
        print("No missing values needed to be filled from GEOS-CF")

    if not silent:
        print("Final merged columns:", merg.columns)

    # Convert timestamps if needed
    all_data = funcs.convert_times_column(merg, 'time', lat, lon)


    return all_data


# convert_no2_mol_m3_to_ppbv is imported from MLpred.read_pandora

def read_validation_set( openaq_id=None, source=None, name=None, url=None, time_col=None, date_format=None, obs_val_col=None, lat_col=None, lon_col=None, species=None, lat=999, lon=-999, unit=None, start=None, end=None, remove_outlier = None, **kwargs, ):

    all_obs = pd.DataFrame()
    isite = ObsSite(
        openaq_id, model_source=None, species=species, observation_source=source
    )
    isite._silent = True
    #print(start)
    isite.read_obs(
        source=source,
        url=url,
        time_col=time_col,
        date_format=date_format,
        value_collum=obs_val_col,
        lat_col=lat_col,
        lon_col=lon_col,
        species=species,
        lat=lat,
        lon=lon,
        unit=unit,
        start=start,
        end=end,
        rename_column = name,
        remove_outlier = remove_outlier,
        **kwargs,
    )

    return isite._obs




def convert_no2_to_ppbv(
    no2_concentration_mol_m3=None,
    volume_m3=1.0,
    pressure_pa=None,
    elevation_m=None,
    temperature_k=None,
):

    ideal_gas_constant = 8.314  # J/(mol·K)

    no2_concentration_mol_cm3 = no2_concentration_mol_m3 * 1.0e-6

    moles_no2 = no2_concentration_mol_cm3 * volume_m3

    pressure_conv = pressure_pa * 100
    pressure_at_sea_level = (
        pressure_conv
        * (1 - 0.0065 * elevation_m / (temperature_k + 0.0065 * elevation_m + 273.15))
        ** 5.2561
    )
    moles_no2_adjusted = moles_no2 * pressure_at_sea_level / pressure_conv

    # Convert adjusted concentration to ppbv
    ppbv = (moles_no2_adjusted / volume_m3) * 1.0e9  # 1 ppbv = 1e9 molecules/m³

    return ppbv


def nsites_by_threshold(df, maxconc=50):
    """Write number of sites with mean concentration above concentration threshold for concentrations ranging from 0 to maxconc ppbv"""
    concrange = np.arange(maxconc + 1) * 1.0
    ns = []
    for ival in concrange:
        nsit = df.loc[df.value > ival].shape[0]
        ns.append(nsit)
    nsites = pd.DataFrame()
    nsites["threshold"] = concrange
    nsites["nsites"] = ns
    return nsites


def plot_deviation_orig(siteRatios, title=None, minval=-30.0, maxval=30.0):
    """Make global map showing deviation betweeen predictions and observations"""
    siteRatios["text"] = [
        "{0:}, Deviation={1:.2f}%".format(i, j)
        for i, j in zip(siteRatios["name"], siteRatios["relChange"])
    ]
    fig = go.Figure(
        data=go.Scattergeo(
            lon=siteRatios["lon"],
            lat=siteRatios["lat"],
            text=siteRatios["text"],
            mode="markers",
            marker=dict(
                size=siteRatios["obs"],
                sizemode="area",
                color=siteRatios["relChange"],
                cmin=minval,
                cmax=maxval,
                colorscale="RdBu",
                autocolorscale=False,
                reversescale=True,
                line_color="rgb(40,40,40)",
                line_width=0.5,
                colorbar_title="NO2 deviation",
            ),
        )
    )
    fig.update_layout(
        title_text="Test",
        showlegend=False,
        height=300,
        geo=dict(landcolor="rgb(217,217,217)"),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )
    return fig


def merge_intervales_with_model(self, confidenceIntervals):
    """Merge intervals with actual model predictions"""
    all_intervals = confidenceIntervals[
        ["time", "value", "upper", "lower", "mid"]
    ].copy()
    all_intervals = all_intervals.resample("1H").mean()
    all_intervals = all_intervals.reset_index()
    all_intervals.columns = ["time", "value", "upper", "lower", "mid"]
    xg_preditions = self.predict()
    model_data = self._mod[["time", "pm25_rh35_gcc"]].copy()

    model_data["time"] = [
        dt.datetime(i.year, i.month, i.day, i.hour, 0, 0) for i in model_data["time"]
    ]
    xg_preditions = xg_preditions.merge(model_data)
    xg_preditions = xg_preditions.set_index("time").resample("1h").mean().reset_index()
    all_intervals = all_intervals.merge(xg_preditions)
    all_intervals = all_intervals.merge(xg_preditions)
    all_intervals = all_intervals.set_index("time").resample("1D").mean()
    return all_intervals

def resample_selected_columns(df, timecol, clmn_grps, resample_freq='1D'):

    df[timecol] = pd.to_datetime(df[timecol])
    

    all_columns = list(set([item for sublist in clmn_grps for item in sublist]))
    

    columns_to_keep = [timecol] + all_columns
    
    df_selected = df[columns_to_keep].copy()

    df_selected.set_index(timecol, inplace=True)
    
    df_resampled = df_selected.resample(resample_freq).mean()
    df_resampled.reset_index(inplace=True)
    
    return df_resampled




def shap_analysis(xgb_model, X_test, selected_features, plot_path, location):
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test[:100])


    shap.summary_plot(shap_values, X_test[:100], plot_type="bar", feature_names=selected_features, show=False)
    plt.title(f"Feature Importance in {location}")
    plt.tight_layout()
    plt.savefig(f'{plot_path}_{location}_feature_importance.png')
    plt.close()

    shap.summary_plot(shap_values, X_test[:100], feature_names=selected_features, show=False)
    plt.title(f"Feature Contribution in {location}")
    plt.tight_layout()
    plt.savefig(f'{plot_path}{location}_feature_contribution.png')
    plt.close()
    
    
def export_to_gesdisc(
    HAQAST_DATA=None,
    location_name=None,
    species=None,
    unit=None,
    start=datetime.today() - relativedelta(years=1),
    end=datetime.today(),
    lat=None,
    lon=None,
    IdentifierProductDOI=None,
):
    """Convert forecasts with GES DISC Formatting"""

    current_datetime = dt.datetime.now()
    current_time_GMT = dt.datetime.utcnow()
    current_time_GMT = time.mktime(current_time_GMT.timetuple())
    HAQAST_DATA["time"] = pd.to_datetime(HAQAST_DATA["time"])
    fvar = "pm25_rh35_gcc" if species == "pm25" else species

    first_timestamp = HAQAST_DATA["time"].min()
    last_timestamp = HAQAST_DATA["time"].max()
    min_year = HAQAST_DATA["time"].dt.year.min()
    max_year = HAQAST_DATA["time"].dt.year.max()
    location_name = location_name
    location_name = location_name.replace(" ", "_")
    lon = lon
    lat = lat
    parameter = species
    unit = unit
    VersionID = "1.0.0"
    Format = "ASCII"
    RangeBeginningDate = first_timestamp.strftime("%Y-%m-%d")
    RangeBeginningTime = first_timestamp.strftime("%H:%M:%S")
    RangeEndingDate = last_timestamp.strftime("%Y-%m-%d")
    RangeEndingTime = last_timestamp.strftime("%H:%M:%S")

    ProductionDateTime = current_datetime.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    ProcessingLevel = "Level 4"
    Conventions = "ASCII"
    DataSetQuality = "A description of the bias-correction methodology and corresponding uncertainty estimates are provided in Keller et al. 2021 (https://doi.org/10.5194/acp-21-3555-2021)"
    title = "HAQAST localized ground-level concentration of nitrogen dioxide (NO2): model-observation fused,1-Hourly,Time-Averaged,Ground-Level (2m)"
    history = f"Original file generated: {current_time_GMT}"
    source = "GEOS-CF v1.0"
    institution = "NASA Global Modeling and Assimilation Office"
    references = "http://gmao.gsfc.nasa.gov"
    TemporalRange = f"{RangeBeginningDate} -> {RangeEndingDate}"
    filename = f"HAQAST_localized_concentration_{parameter}.L4.V1.{location_name}.{min_year}-{max_year}.txt"
    StationLatitude = lat
    StationLongitude = lon
    Contact = "http://gmao.gsfc.nasa.gov"
    (
        SouthernmostLatitude,
        NorthernmostLatitude,
        WesternmostLongitude,
        EasternmostLongitude,
    ) = calculate_extremes([(lat, lon)])
    SpatialCoverage = "point-source"

    if species == "no2":
        IdentifierProductDOI = "10.5067/R3MOD87DBR3E"
        shortName = "HAQLOCNO2"
    elif species == "pm25":
        IdentifierProductDOI = "10.5067/MGBETJN7JJCS"
        shortName = "HAQLOCPM25"
    elif species == "o3":
        IdentifierProductDOI = "10.5067/11JBPNUERB7L"
        shortName = "HAQLOCO3"
    else:
        IdentifierProductDOI = "10.5067/R3MOD87DBR3E"
        shortName = "HAQLOCNO2"

    HAQAST_DATA = HAQAST_DATA.rename(
        columns={
            "time": "ISO8601",
            "localised": "localized_model_value",
            fvar: "uncorrected_model_value",
        }
    )
    HAQAST_DATA["ISO8601"] = pd.to_datetime(HAQAST_DATA["ISO8601"]).dt.strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    HAQAST_DATA["location"] = location_name
    HAQAST_DATA["lat"] = lat
    HAQAST_DATA["lon"] = lon
    HAQAST_DATA["parameter"] = parameter
    HAQAST_DATA["unit"] = unit

    HAQAST_DATA = HAQAST_DATA[
        [
            "ISO8601",
            "location",
            "lat",
            "lon",
            "parameter",
            "unit",
            "localized_model_value",
            "uncorrected_model_value",
        ]
    ]

    # prepare HAQAST Metadata
    HAQAST_DATA["ISO8601"] = pd.to_datetime(HAQAST_DATA["ISO8601"], errors="coerce")

    folder_path = "HAQAST_localized_concentration_L4"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = os.path.join(folder_path, filename)

    with open(file_path, "a") as file:
        file.truncate(0)
        file.write(
            "#######################################################################\n"
        )
        file.write(f'## GranuleID = "{filename}" \n')
        file.write(f'## ShortName = "{shortName} \n')
        file.write(f'## DOI = "{IdentifierProductDOI}S \n')
        file.write(
            f'## LongName = "HAQAST localized ground-level concentration of {parameter}: model-observation fused,1-Hourly,Time-Averaged,Ground-Level (2m)" \n'
        )
        file.write(f'## VersionID = "{VersionID}" \n')
        file.write(f'## Format = "{Format}" \n')
        file.write(f'## VersionID = "{VersionID}" \n')
        file.write(f'## RangeBeginningDate = "{RangeBeginningDate}" \n')
        file.write(f'## RangeBeginningTime = "{RangeBeginningTime}" \n')
        file.write(f'## RangeEndingDate = "{RangeEndingDate}" \n')
        file.write(f'## RangeEndingTime = "{RangeEndingTime}" \n')
        file.write(f'## IdentifierProductDOI = "{IdentifierProductDOI}" \n')
        file.write(f'## ProductionDateTime = "{ProductionDateTime}" \n')
        file.write(f'## ProcessingLevel = "{ProcessingLevel}" \n')
        file.write(f'## Conventions = "{Conventions}" \n')
        file.write(f'## DataSetQuality = "{DataSetQuality}" \n')
        file.write(f'## Title = "{title}" \n')
        file.write(f'## History = "{history}" \n')
        file.write(f'## Source = "{source}" \n')
        file.write(f'## Institution = "{institution}" \n')
        file.write(f'## references = "{references}" \n')
        file.write(f'## TemporalRange = "{TemporalRange}" \n')
        file.write(f'## Filename = "{filename}" \n')
        file.write(f'## StationLatitude = "{StationLatitude}" \n')
        file.write(f'## StationLongitude = "{StationLongitude}" \n')
        file.write(f'## Contact = "{Contact}" \n')
        file.write(f'## SouthernmostLatitude = "{SouthernmostLatitude}" \n')
        file.write(f'## NorthernmostLatitude = "{NorthernmostLatitude}" \n')
        file.write(f'## WesternmostLongitude = "{WesternmostLongitude}" \n')
        file.write(f'## EasternmostLongitude = "{EasternmostLongitude}" \n')
        file.write(f'## SpatialCoverage = "{SpatialCoverage}" \n')
        file.write(
            "#######################################################################\n"
        )
        HAQAST_DATA.to_csv(file, index=False)
        print("dataproduct is saved successfully ")


def get_site_information(site_id):
    """Collect location information from OpenAQ"""
    url = (
        "https://api.openaq.org/v2/locations/"
        + str(site_id)
        + "?limit=100&page=1&offset=0&sort=desc&radius=1000&order_by=lastUpdated&dumpRaw=false"
    )

    headers = {"accept": "application/json"}

    response = requests.get(url, headers=headers).json()

    return response["results"][0]["name"]


def plot_actual_vs_predicted(actual_values, predicted_values, xlabel, ylabel, title, file_name, color='blue'):
    actual_values = np.array(actual_values)
    predicted_values = np.array(predicted_values)
    
    mask = np.isfinite(actual_values) & np.isfinite(predicted_values)
    actual_values = actual_values[mask]
    predicted_values = predicted_values[mask]
    
    if len(actual_values) == 0 or len(predicted_values) == 0:
        print("No valid data points to plot.")
        return
    
    X = actual_values.reshape(-1, 1)  
    y = predicted_values

    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0]
    intercept = model.intercept_

    plt.scatter(actual_values, predicted_values, label='Estimated sfcmr', color=color, marker='o') 
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.plot(actual_values, model.predict(X), color='black', linewidth= 0.5, label=f'y = {slope:.2f}x + {intercept:.2f}')

   
    plt.plot(actual_values, actual_values, 'r--')
    
    max_val = max(np.max(actual_values), np.max(predicted_values))
    min_val = min(np.min(actual_values), np.min(predicted_values))

    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    plt.grid(True)
    plt.legend() 
    plt.savefig(f'plots/pandora_cf/{file_name}.png', dpi=1200)
    plt.show()
    

def scatter_plot_multiple_markers(df, x_column, y_columns, markers=None, title=None, xlabel=None, ylabel=None, marker_fill=False, file_name=None, directory = "plots/pandora_cf/"):
    colors = {
        "corrected_openaq": "orange",
        "openaq": "green",
        "corrected_pandora": "red",
        "no2": "black",
        "pandora": "blue",
        "estimated_sfcmr": "red",
    }

    if markers is None:
        markers = ['o', 's', '^', 'D', 'x', '+', '*', 'p', 'h', 'v']

    if not isinstance(x_column, str):
        raise ValueError("x_column must be a string.")

    if x_column not in df.columns:
        raise ValueError(f"x_column '{x_column}' not found in DataFrame.")

    if not all(isinstance(col, str) for col in y_columns):
        raise ValueError("All elements in y_columns must be strings.")

    valid_y_columns = [col for col in y_columns if col in df.columns]

    if not valid_y_columns:
        raise ValueError("No valid y_columns found in DataFrame.")

    fig, ax = plt.subplots()

    for i, y_col in enumerate(valid_y_columns):
        marker = markers[i % len(markers)]
        color = colors.get(y_col, 'b')
        
        if y_col == 'local_obs':
            marker_label ="Local Observation"
        elif y_col == 'openaq':
            marker_label ="OpenAQ"
        elif y_col == 'no2':
            marker_label ="GEOS-CF"
        elif y_col == 'value':
            marker_label ="Pandora"
        elif y_col == 'estimated_sfcmr':
            marker_label = "Hybrid (Pandora-CF)"
        else:
            marker_label = y_col
            
                

        if marker_fill:
            ax.scatter(df[x_column], df[y_col], marker=marker, color=color, label=marker_label)
        else:
            ax.scatter(df[x_column], df[y_col], marker=marker, edgecolors=color, facecolors='none', label=marker_label)


        data_min = min(df[x_column].min(), df[y_col].min())
        data_max = max(df[x_column].max(), df[y_col].max())
        ax.set_xlim(data_min, data_max)
        ax.set_ylim(data_min, data_max)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_aspect('equal', adjustable='box')

    plt.grid(True)
    plt.plot(df[x_column], df[x_column], color='black', linewidth=0.5)
    plt.legend()
    plt.savefig(f'{directory}/{file_name}_scatter_plot_{xlabel}.png', dpi=1200)
    print("scatter plot saved")
    plt.show()
    

    
def evaluate_time_series(actual, predicted):
    
    if len(actual) == 0 or len(predicted) == 0:
        print("No valid data points to plot.")
        return
    
    actual = np.array(actual)
    predicted = np.array(predicted)
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    actual = actual[mask]
    predicted = predicted[mask]


    residuals = actual - predicted

    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    percentage_within_threshold = np.mean(np.abs(residuals) <= 0.1 * np.max(actual)) * 100  
    pearson_corr, _ = pearsonr(actual, predicted)
    spearman_corr, _ = spearmanr(actual, predicted)
    r_squared = r2_score(actual, predicted)
    autocorr_residuals = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    std_dev = np.std(residuals)
    mean_bias = np.mean(residuals)
    mean_actual = np.mean(actual)
    mean_predicted = np.mean(predicted)

    results = pd.DataFrame({
        'Statistic': ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error',
                      'Mean Absolute Percentage Error', 'Percentage Within Threshold',
                      'Pearson Correlation', 'Spearman Correlation', 'R-squared',
                      'Autocorrelation of Residuals', 'Standard Deviation of Residuals',
                      'Mean Bias', 'Mean Actual Value', 'Mean Predicted Value'],
        'Value': [mae, mse, rmse, mape, percentage_within_threshold,
                  pearson_corr, spearman_corr, r_squared, autocorr_residuals,
                  std_dev, mean_bias, mean_actual, mean_predicted]
    })

    return results
    
    
    
def calculate_extremes(coordinates):
    """Calculate coordinate extremes based on lat, lon of the location"""
    southernmost_lat = min(coordinates, key=lambda x: x[0])[0]
    northernmost_lat = max(coordinates, key=lambda x: x[0])[0]
    westernmost_long = min(coordinates, key=lambda x: x[1])[1]
    easternmost_long = max(coordinates, key=lambda x: x[1])[1]

    return southernmost_lat, northernmost_lat, westernmost_long, easternmost_long


def convert_pollutant(species, value, current_unit, conversion_unit):
    """Unit conversions routine"""
    
    PPB2UGM3 = {"no2": 1.88, "o3": 1.97}
    VVtoPPBV = 1.0e9

    if current_unit == "ugm3" and conversion_unit == "ppbv":
        print(f'converting from {current_unit} to {conversion_unit}')
        value = value * (1.0 / PPB2UGM3[species])
            
    if current_unit == conversion_unit:
        value = value
    
    return value 


def convert_ugm3_to_ppbv(value,species):

    PPB2UGM3 = {"no2": 1.88, "o3": 1.97}
    VVtoPPBV = 1.0e9
    conv_factor = PPB2UGM3[species]
    ppbv = value ** 1.0 / conv_factor

    return ppbv



def get_location_info(url, location_code):
    print(url)
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if location_code in data:
            location_data = data[location_code]
            info = {}
            
            for key, value in location_data.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, dict):
                            for sub_sub_key, sub_sub_value in sub_value.items():
                                info[f'{key}_{sub_key}_{sub_sub_key}'] = sub_sub_value
                        else:
                            info[f'{key}_{sub_key}'] = sub_value
                else:
                    info[key] = value
            
            return info
        else:
            print(f"Location with code {location_code} not found")
            return None
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None
    
    

def get_no2_locations(lat, lon, radius_km):

    endpoint = 'https://api.openaq.org/v2/locations'
    

    params = {
        'latitude': lat,
        'longitude': lon,
        'radius': radius_km * 1000,  
        'parameter': 'no2',
        'limit': 1000
    }
    
    headers = {'X-API-Key': '849b4d760f2679b9e0cafea2a23a698578107984fb150a81d15640600bdb271f'}
    response = requests.get(endpoint, params=params, headers= headers)
    
    if response.status_code == 200:
        data = response.json()

        return [{'id': location['id'], 'name': location['name'], 'coordinates': location['coordinates']} for location in data['results']]
    else:
        print('Error:', response.status_code, response.text)
        return []

def process_location_data(code=None):
    """
    Process location-specific NO2 datas and generate visualizations.
    
    Args:
        code (str): Location identifier
    """

    location = mlpred.get_location_info("https://raw.githubusercontent.com/noussairlazrak/MLpred/refs/heads/main/global.json", code)
    if not location:
        print("Location not found.")
        return

    config = {
        'location_name': location['location_name'],
        'species': 'no2',
        'lat': location['lat'],
        'lon': location['lon'],
        'model_source': 'local',
        'observation_source': 'pandora',
        'obs_url': location['obs_options_no2_file'],
        'resample': '1h',
        'unit': location['obs_options_no2_unit'],
        'model_date_col' : 'time'
    }
    
    location_name = config["location_name"]
    airnow_sensor_ids = []

    if "openaq" in data[code] and "o3" in data[code]["openaq"]:
        
        for item in data[code]["openaq"]["o3"]:
            if isinstance(item, dict) and "provider" in item and isinstance(item["provider"], dict) and "name" in item["provider"] and item["provider"]["name"] == "AirNow" and "sensors" in item:
                sensors = item["sensors"]
                if isinstance(sensors, list): 
                    for sensor in sensors:
                        if isinstance(sensor, dict) and "id" in sensor: 
                            airnow_sensor_ids.append(sensor["id"])

    
        
    isite = mlpred.ObsSite(
        None, 
        model_source=config['model_source'], 
        species=config['species'], 
        observation_source=config['observation_source']
    )
    isite._silent = False
    resample_size = "1h"

   
    isite.read_obs(
        source=config['observation_source'], 
        url=config['obs_url'], 
        time_col=config['unit'], 
        date_format='%Y-%m-%d %H:%M',
        value_collum='value', 
        lat_col='lat', 
        lon_col='lon', 
        species=config['species']
    )
    isite.read_mod(source=config['model_source'], url='./geos_cf/model_demo.csv', model_date_col= "time")
    
    merged_data = isite._merge(interpolation=True)
    
    print(merged_data.columns)
    print(merged_data[["no2","pandora","time"]])
    
    merged_data.dropna(inplace=True)
    
    #print(merged_data.columns)

    return config, merged_data, isite._mod, isite._obs, airnow_sensor_ids

def process_location_data(code=None):
    """
    Process location-specific NO2 data and generate visualizations.
    
    Args:
        code (str): Location identifier
    """

    location = get_location_info("https://www.noussair.com/global.json", code)
    if not location:
        print("Location not found.")
        return

    config = {
        'location_name': location['location_name'],
        'species': 'no2',
        'lat': location['lat'],
        'lon': location['lon'],
        'model_source': 's3',
        'observation_source': 'pandora',
        'openaq_id': location['openaq'],
        'obs_url': location['obs_options_no2_file'],
        'resample': '1h',
        'unit': location['obs_options_no2_unit']
    }
    location_name = config["location_name"]
    isite = ObsSite(
        config['openaq_id'], 
        model_source=config['model_source'], 
        species=config['species'], 
        observation_source=config['observation_source']
    )
    isite._silent = True
    resample_size = "1D"

    # Read observations
    isite.read_obs(
        source=config['observation_source'], 
        url=config['obs_url'], 
        time_col=config['unit'], 
        date_format='%Y-%m-%d %H:%M',
        value_collum='no2', 
        lat_col='lat', 
        lon_col='lon', 
        species=config['species']
    )

    
    isite.read_mod(source=config['model_source'], url=None)
    merged_data = isite._merge(interpolation=True)
    merged_data.dropna(inplace=True)


    
    openaq_obs = read_validation_set(
        source='openaq', 
        openaq_id=config['openaq_id'], 
        name='Openaq', 
        species='no2', 
        time_col='date', 
        date_format='%Y-%m-%d %H:%M',
        start=dt.datetime(2018, 1, 1), 
        end=dt.datetime.today(), 
        unit='ppbv'
    )
    
    if openaq_obs.empty:
        openaq_obs = pd.DataFrame(columns=['date', 'openaq'])
        preddf = merged_data.copy()

    if not openaq_obs.empty:
        openaq_obs.rename(columns={'value': 'openaq'}, inplace=True)
        preddf = funcs.merge_dataframes(
        [merged_data, openaq_obs[["time","openaq"]]], 
        index_col="time", 
        resample=f'{resample_size}', 
        how="outer"
    )
    
    
    final_data = xgboost_bias_correction(preddf)
    
    #final_data.to_csv(f'./pandora_cf/results_dataframes/{location_name}.csv')
    
    final_data['time'] = pd.to_datetime(final_data['time'])

    final_data.set_index('time', inplace=True)
    
    

    resfin = final_data[["no2", "pandora", "openaq", "corrected_pandora"]].resample('5D').mean()

    resfin = resfin.reset_index()
    final_data = final_data.reset_index()
    

    if gen_viz:

        try:
            location_plot(
            dataframe=resfin,
            location_name=location_name,
            title=f'{location_name} (10 avg)',
            species='no2',
            unit='ppbv',
            resample='10D',
            directory = "plots/pandora_cf/pandora_vs_cf/"
        )

        except Exception as general_error:
            print(f"Unexpected ploting the timeseries plot in {location_name}")


        try:
            scatter_plot_multiple_markers(final_data, 'openaq', ["no2", "pandora","corrected_pandora","corrected_openaq"], markers=None, title=f'{config["location_name"]} ({resample_size})', xlabel='OpenAQ (ppbv)', ylabel='Surface concentration (ppbv)', file_name=f'{config["location_name"]}_{resample_size}', directory = "plots/pandora_cf/pandora_vs_cf/")

        except Exception as general_error:
                print(f"Unexpected ploting the scatter plot in {location_name}")
        
    
    locations = [
        {
            'name': 'Pandora', 
            'lat': final_data["lat"].mean(), 
            'lon': final_data["lon"].mean(), 
            'type': 'Pandora'
        }
       
    ]
    
    openaq_locations = get_no2_locations(config["lat"], config["lon"], 10)
        
    for loc in openaq_locations:
        locations.append({
            'name': loc['name'],
            'lat': loc['coordinates']['latitude'],
            'lon': loc['coordinates']['longitude'],
            'type': 'OpenAQ'
        })

    return locations


def get_openaq_locations(api_key=OPENAQAPI, lat=None, lon=None, radius=None, parameter=None, max_retries=5, base_delay=1):
    """Retrieves OpenAQ monitoring locations with a specific parameter (e.g., 'no2') and their details."""
    import requests
    import time

    API_BASE_URL = "https://api.openaq.org/v3/locations"
    params = {"coordinates": f"{lat},{lon}" if lat and lon else None, "radius": radius, "limit": 1000}
    # Remove None values from params
    params = {k: v for k, v in params.items() if v is not None}
    headers = {"Accept": "application/json", "X-API-Key": api_key} if api_key else {"Accept": "application/json"}

    locations_data = []
    for attempt in range(max_retries):
        try:
            res = requests.get(API_BASE_URL, params=params, headers=headers)
            res.raise_for_status()
            locations = res.json().get("results", [])
            print()

            for loc in locations:
                filtered_sensors = [
                    sensor for sensor in loc.get("sensors", [])
                    if sensor.get("parameter", {}).get("name") == parameter
                ]
                if filtered_sensors:
                    locations_data.append({
                        "location_id": loc["id"],
                        "name": loc.get("name") or loc.get("location"),
                        "provider": loc.get("provider"),
                        "coordinates": loc.get("coordinates"),
                        "sensors": [{
                            "id": sensor["id"],
                            "coordinates": sensor.get("coordinates")
                        } for sensor in filtered_sensors]
                    })
            return locations_data
        except requests.RequestException as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(base_delay * (2 ** attempt))
    return locations_data


def get_location_sensors(location_id, api_key=OPENAQAPI, parameter=None, max_retries=5, base_delay=1):
    """Retrieves sensor data for a specific location and parameter."""
    SENSOR_API_URL = f"https://api.openaq.org/v3/locations/{location_id}/sensors"
    headers = {"Accept": "application/json", "X-API-Key": api_key} if api_key else {"Accept": "application/json"}
    params = {"parameter": parameter}

    for attempt in range(max_retries):
        try:
            res = requests.get(SENSOR_API_URL, params=params, headers=headers)
            res.raise_for_status()
            return res.json().get("results", [])
        except requests.exceptions.HTTPError as e:
            if res.status_code == 429:
                time.sleep(base_delay * (2 ** attempt) + random.random())
            else:
                print(f"HTTP Error. {e}")
                break
        except Exception as e:
            print(f"Error. {e}")
            break

    print(f"Failed to retrieve sensor data for location {location_id}")
    return []

def pandora_maps(url, showtext = False, save_path='./plots', title =None, srcs = None):
    try:
        with urllib.request.urlopen(url) as resp:
            dat = json.load(resp)
    except urllib.error.URLError as e:
        print(f"Error opening URL: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return

    fig = plt.figure(figsize=(12, 6), dpi=300)
    mp = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    mp.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    mp.add_feature(cfeature.COASTLINE, linewidth=0.5)
    mp.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    mp.add_feature(cfeature.LAND, facecolor='whitesmoke')
    mp.add_feature(cfeature.OCEAN, facecolor='aliceblue')

    pan_ptch = None
    opn_ptch = None
    air_ptch = None

    for loc_id, loc_dat in dat.items():
        loc_nm = loc_dat.get("location_name", "Unknown")
        lat = loc_dat.get("lat")
        lon = loc_dat.get("lon")
        opn_data = loc_dat.get("openaq", {}).get("o3", [])

        if lat and lon:
            pan_mrkr = mp.plot(lon, lat, marker='o', markersize=15, transform=ccrs.PlateCarree(),
                                markeredgecolor='black', markerfacecolor='none', markeredgewidth=0.3,
                                label='Pandora' if pan_ptch is None else "")
        if showtext:
            if len(opn_data) > 0:
                bbox_props = dict(boxstyle="square,pad=0.3", facecolor="red", alpha=1, edgecolor="none")
                text = f"{loc_nm} \n Nearby OpenAQ: {len(opn_data)}"

                mp.annotate(text, xy=(lon, lat), xycoords=ccrs.PlateCarree()._as_mpl_transform(mp),
                            xytext=(30, 30), textcoords='offset points',
                            bbox=bbox_props, color="white", fontsize=8,
                            arrowprops=dict(arrowstyle="-", color="black", linewidth=1, shrinkA=0, shrinkB=0,
                                            connectionstyle="arc3,rad=0"))
    
        for sta in opn_data:
            sta_nm = sta.get("name", "Unknown Station")
            coords = sta.get("coordinates")
            prov = sta.get("provider", {}).get("name")

            if coords:
                sta_lat = coords.get("latitude")
                sta_lon = coords.get("longitude")

                if sta_lat and sta_lon:
                    if prov == "AirNow":
                        air_mrkr = mp.plot(sta_lon, sta_lat, marker='s', markersize=5, transform=ccrs.PlateCarree(),
                                        markeredgecolor='green', markerfacecolor='none', markeredgewidth=0.3,
                                        label='AirNow' if air_ptch is None else "")
                        if air_ptch is None:
                            air_ptch = air_mrkr[0]
                    else:
                        opn_mrkr = mp.plot(sta_lon, sta_lat, marker='^', markersize=5, transform=ccrs.PlateCarree(),
                                        markeredgecolor='blue', markerfacecolor='none', markeredgewidth=0.3,
                                        label='OpenAQ' if opn_ptch is None else "")
                        if opn_ptch is None:
                            opn_ptch = opn_mrkr[0]

        if pan_ptch is None and pan_mrkr:
            pan_ptch = pan_mrkr[0]

    handles = [pan_ptch, opn_ptch, air_ptch]
    labels = ['Pandora', 'OpenAQ', 'AirNow']

    filtered_handles = [h for h in handles if h]
    filtered_labels = [labels[i] for i, h in enumerate(handles) if h]

    if filtered_handles:
        plt.legend(handles=filtered_handles, labels=filtered_labels, loc='upper left')

    plt.title("")

    try:
        font_family = 'Arial'
        font = fm.FontProperties(family=font_family, fname=fm.findfont(font_family), size=16, weight='bold')

        fig.suptitle(title,
                     fontsize=20, fontweight='bold', color='black', fontproperties=font, ha='left', x=0.125, y = 0.92)

        font_subtitle = fm.FontProperties(family=font_family, fname=fm.findfont(font_family), size=10, weight = 'bold')

        fig.text(0.125, 0.86, srcs,
                 ha='left', va='top', fontsize=10, color='grey', fontproperties=font_subtitle)

        plt.subplots_adjust(top=0.8)

    except Exception as e:
        print(f"Error setting font: {e}. Using default font.")


    plt.savefig(save_path, dpi=300)
    plt.show()
    
def read_geos_cf(lon, lat, start=None, end=None, version=2, use_cache=True, verbose=True):
    """
    Read GEOS-CF model data from S3 with caching support.
    
    Caching Strategy:
    - Saves replay data (historical) to CSV: GEOS_CF/loc_{lat}_{lon}_v{version}.csv
    - On subsequent calls: loads cached replay, only fetches new analysis/forecast
    - Significantly reduces S3 read time (replay is ~6+ years of data)
    
    Parameters
    ----------
    lon, lat : float
        Location coordinates
    start, end : datetime, optional
        Date range filter
    version : int, default=2
        GEOS-CF version (1 or 2)
    use_cache : bool, default=True
        If True, uses cached replay data and only fetches new analysis/forecast
    verbose : bool, default=True
        If True, prints progress messages
    
    Returns
    -------
    pandas.DataFrame
        Data with species in ppbv and time features
    
    Examples
    --------
    >>> # First call: reads all data from S3, saves replay to cache
    >>> df = read_geoscf(lon=-77.0, lat=38.9, version=2)
    
    >>> # Subsequent calls: loads replay from cache, only fetches new data
    >>> df = read_geoscf(lon=-77.0, lat=38.9, version=2)  # Much faster!
    """
    # Create cache directory if it doesn't exist
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
    
    # filename
    lat_str = f"{lat:.6f}".replace('.', '_').replace('-', 'm')
    lon_str = f"{lon:.6f}".replace('.', '_').replace('-', 'm')
    cache_file = os.path.join(CACHE_DIR, f"loc_{lat_str}_{lon_str}_v{version}.csv")
    
    # S3 paths
    paths = {
        1: [S3_TEMPLATE, S3_REPLAY_TEMPLATE, S3_FORECASTS_TEMPLATE],
        2: [S3_V2_RPL, S3_V2_COLS, S3_V2_FCST]
    }[version]
    
    df_replay = None
    
    # load cached replay data
    if use_cache and os.path.exists(cache_file):
        try:
            if verbose:
                print(f"Loading cached replay data from {cache_file}")
            df_replay = pd.read_csv(cache_file, parse_dates=['time'])
            if verbose:
                print(f"Loaded {len(df_replay)} cached rows (from {df_replay['time'].min()} to {df_replay['time'].max()})")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not load cache file: {e}. Will fetch from S3.")
            df_replay = None
    
    # Read all sources
    dfs = []
    

    if df_replay is not None:
        dfs.append(df_replay)
        start_index = 2 if version == 2 else 1  # Skip replay (and replay_cols for V2)
        if verbose:
            print(f"Skipping replay data fetch, only fetching recent data...")
    else:
        start_index = 0
    
    for i, path in enumerate(paths):
        # Skip replay bucket
        if i < start_index:
            continue
        
        source_name = ["replay", "replay_cols", "forecast"][i] if version == 2 else ["replay", "analysis", "forecast"][i]
        
        if verbose:
            print(f"Reading {source_name} from S3...")
        
        try:
            ds = xr.open_zarr(fsspec.get_mapper(path), consolidated=True)
            sel = {"lon": lon, "lat": lat, "method": "nearest"}
            if "lev" in ds.dims or "lev" in ds.coords:
                sel["lev"] = 1
            df = ds.sel(**sel).load().to_dataframe().reset_index()
            df["time"] = pd.to_datetime(df["time"])
            
            if verbose:
                print(f"Read {len(df)} rows from {source_name}")
            
            # V2: merge replay and replay_cols horizontally
            if version == 2 and i == 1 and dfs and df_replay is None:
                # Only do horizontal merge if we're reading both replay sources
                dfs[0] = pd.merge(dfs[0], df, on="time", how="outer", suffixes=("", "_x"))
                dfs[0] = dfs[0][[c for c in dfs[0].columns if not c.endswith("_x")]]
                continue
            
            dfs.append(df)
        except Exception as e:
            if verbose:
                print(f"Error reading {source_name}: {e}")
    
    if not dfs:
        if verbose:
            print("[ERROR] No data could be retrieved")
        return pd.DataFrame()
    
    # Merge vertical
    if verbose:
        print(f"Combining {len(dfs)} data sources...")
    
    df = pd.concat(dfs, ignore_index=True).sort_values("time").drop_duplicates("time", keep="first").reset_index(drop=True)
    
    # Save replay data to cache
    if use_cache and df_replay is None:
        try:
            # V1: save replay only
            # V2: save replay + replay_cols merged 
            if version == 1:
                # V2 Replay 
                replay_times = df[df['time'] <= (datetime.now() - pd.Timedelta(days=30))]
            else:
                # V2, replay+cols
                replay_times = df[df['time'] <= (datetime.now() - pd.Timedelta(days=5))]
            
            if len(replay_times) > 0:
                replay_times.to_csv(cache_file, index=False)
                if verbose:
                    print(f"Saved {len(replay_times)} replay rows to {cache_file}")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not save cache file: {e}")
    
    # Harmonization
    df = df.rename(columns={"t10m": "t", "u10m": "u", "v10m": "v", "pm25_rh35_gcc": "pm25_rh35"})
    
    # Derived AOD
    if "aod550_sala" in df.columns and "aod550_salc" in df.columns:
        df["aod550_ss"] = df["aod550_sala"] + df["aod550_salc"]
    
    # Filter dates
    if start:
        df = df[df["time"] >= start]
    if end:
        df = df[df["time"] <= end]
    
    # Convert to ppbv
    for sp in DEFAULT_GASES + ["so2"]:  # Add so2 to the DEFAULT_GASES list
        if sp in df.columns:
            df[sp] *= VVtoPPBV
    
    # Add time features
    df["month"] = df["time"].dt.month
    df["hour"] = df["time"].dt.hour
    df["weekday"] = df["time"].dt.weekday
    
    if verbose:
        print(f"Retrieved {len(df)} total rows from {df['time'].min()} to {df['time'].max()}")
    
    return df.reset_index(drop=True)