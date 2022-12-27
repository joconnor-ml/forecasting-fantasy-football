import json
import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
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
            for alpha in [0.1, 1, 10, 100]
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
        return (df["season"] == utils.SEASONS[-1]) & (
            df["GW"]
            == df[(df["season"] == utils.SEASONS[-1]) & df["total_points"].notnull()][
                "GW"
            ].max()
        )

    def train_test_split(self, df, features, targets):
        # predicting scores conditioned on player appearing
        train_filter = df["season"].isin(utils.SEASONS[:-2])
        val_filter = df["season"].isin(utils.SEASONS[-2:])
        top_val_filter = df["season"].isin(utils.SEASONS[-2:]) & (
            df["selected_by_percent"] > 10
        )

        train_features = features[train_filter]
        val_features = features[val_filter]
        top_val_features = features[top_val_filter]
        train_targets = targets[train_filter][TARGET_COL]
        val_targets = targets[val_filter][TARGET_COL]
        top_val_targets = targets[top_val_filter][TARGET_COL]
        return (
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

    def train(self, train_features, train_targets, **fit_kwargs):
        self.feature_names = train_features.columns
        self.model = self.model.fit(
            train_features, self.transform(train_targets), **fit_kwargs
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
                utils.generate_targets(df, self.horizon, ["was_home"])
                .fillna(False)
                .astype("Int32")
                .astype(float),
                utils.generate_targets(df, self.horizon, ["win_prob"]),
                utils.generate_rolling_features(
                    df,
                    ["minutes", "xP"],
                    aggs=("mean",),
                ),
                utils.generate_lag_features(df, ["minutes", "xP"], lags=(0,)),
                (
                    utils.generate_lag_features(df, ["value_rank"], lags=(0,))
                    + np.random.uniform(0, 1)
                ).clip(1, 6),
            ],
            axis=1,
        )

    def get_feature_importances(self):
        return pd.Series(self.model.steps[-1][-1].coef_, index=self.feature_names)


class GKModel(PointsModel):
    def generate_features(self, df):
        return pd.concat(
            [
                utils.generate_targets(df, self.horizon, ["was_home"])
                .fillna(False)
                .astype("Int32")
                .astype(float),
                utils.generate_targets(df, self.horizon, ["win_prob"]),
                utils.generate_rolling_features(
                    df, ["saves", "minutes"], aggs=("mean",)
                ),
                utils.generate_lag_features(df, ["value_rank"], lags=(0,)),
            ],
            axis=1,
        )
