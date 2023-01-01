import json
import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import utils

TARGET_COL = "total_points"


def get_models(position, horizon):
    if position == "GK":
        model_class = GKModel
    else:
        model_class = PointsModel
    return {
        **{
            f"lasso_{alpha}": model_class(
                model=make_pipeline(
                    SimpleImputer(), StandardScaler(), Lasso(alpha=alpha)
                ),
                horizon=horizon,
            )
            for alpha in [0.001, 0.01, 0.1, 1]
        },
        **{
            "linear_regression": model_class(
                model=make_pipeline(
                    SimpleImputer(), StandardScaler(), LinearRegression()
                ),
                horizon=horizon,
            )
        },
    }


class PointsModel:
    feature_names = None

    def __init__(self, model, horizon):
        self.model = model
        self.horizon = horizon

    def save_model(self, data, path):
        p = pathlib.Path(path)
        if not p.exists():
            p.mkdir(exist_ok=True, parents=True)
        with open(p / "data.json", "wt") as f:
            json.dump(fp=f, obj=data)
        joblib.dump(self.model, p / "model.joblib")

    def get_targets(self, df):
        return utils.generate_targets(
            df, self.horizon, ["total_points", "minutes", "was_home"]
        )

    def train_filter(self, df, targets):
        return targets["total_points"].notnull() & (df["selected_by_percent"] > 1)

    def inference_filter(self, df, targets):
        # TODO automate getting the inference week
        return (df["season"] == utils.TRAIN_SEASONS[-1]) & (
            df["GW"]
            == df[
                (df["season"] == utils.TRAIN_SEASONS[-1]) & df["total_points"].notnull()
            ]["GW"].max()
        )

    def train_test_split(self, df, features, targets):
        # predicting scores conditioned on player appearing
        train_filter = df["season"].isin(utils.TRAIN_SEASONS[:-2])
        val_filter = df["season"].isin(utils.TRAIN_SEASONS[-2:])
        top_val_filter = df["season"].isin(utils.TRAIN_SEASONS[-2:]) & (
            df["selected_by_percent"] > 10
        )

        train_df = df[train_filter]
        val_df = df[val_filter]
        top_val_df = df[top_val_filter]
        train_features = features[train_filter]
        val_features = features[val_filter]
        top_val_features = features[top_val_filter]
        train_targets = targets[train_filter][TARGET_COL]
        val_targets = targets[val_filter][TARGET_COL]
        top_val_targets = targets[top_val_filter][TARGET_COL]
        return (
            train_df,
            val_df,
            top_val_df,
            train_features,
            val_features,
            top_val_features,
            train_targets,
            val_targets,
            top_val_targets,
        )

    def transform(self, targets):
        return np.clip(targets, 0, np.inf) ** 1.5

    def inverse(self, targets):
        return np.clip(targets, 0, np.inf) ** 0.66

    def train(self, train_features, train_targets, weights=None, **fit_kwargs):
        if weights is not None:
            model_name = self.model.steps[-1][0]
            model_args = {f"{model_name}__sample_weight": weights}
        else:
            model_args = {}
        self.feature_names = train_features.columns
        self.model = self.model.fit(
            train_features, self.transform(train_targets), **model_args, **fit_kwargs
        )
        return self

    def predict(self, test_features):
        return self.inverse(self.model.predict(test_features))

    def get_scores(self, targets, preds):
        return {
            "rmse": mean_squared_error(targets, preds) ** 0.5,
            "mae": mean_absolute_error(targets, preds),
        }

    def generate_features(self, df):
        return pd.concat(
            [
                utils.generate_targets(df, self.horizon, ["elo_diff"]),
                self.transform(
                    utils.generate_rolling_features(
                        df, ["xP"], aggs=("mean",), windows=(5,)
                    )
                ),
                self.transform(
                    utils.generate_rolling_features(
                        df, ["total_points"], aggs=("mean",), windows=(19,)
                    )
                ),
                # (
                #    utils.generate_lag_features(df, ["value_rank"], lags=(0,))
                #    + np.random.uniform(0, 1)
                # ).clip(1, 3),
            ],
            axis=1,
        )

    def get_feature_importances(self):
        return pd.Series(self.model.steps[-1][-1].coef_, index=self.feature_names)


class GKModel(PointsModel):
    def generate_features(self, df):
        return pd.concat(
            [
                utils.generate_targets(df, self.horizon, ["elo_diff"]),
                utils.generate_rolling_features(
                    df, ["saves", "minutes"], aggs=("mean",)
                ),
                # utils.generate_lag_features(df, ["value_rank"], lags=(0,)),
            ],
            axis=1,
        )
