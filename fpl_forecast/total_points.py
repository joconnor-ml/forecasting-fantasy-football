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
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from . import utils

TARGET_COL = "total_points"


def get_models():
    return {
        **{
            f"lasso_{alpha}": make_pipeline(
                SimpleImputer(), StandardScaler(), Lasso(alpha=alpha)
            )
            for alpha in [0.1, 1, 10, 100]
        },
        **{
            f"lasso_poly_{alpha}": make_pipeline(
                SimpleImputer(),
                PolynomialFeatures(),
                StandardScaler(),
                Lasso(alpha=alpha),
            )
            for alpha in [0.1, 1, 10]
        },
    }


def save_model(model, data, path):
    p = pathlib.Path(path)
    if not p.exists(): p.mkdir(exist_ok=True, parents=True)
    with open(p / "data.json", "wt") as f:
        json.dump(fp=f, obj=data)
    joblib.dump(model, p / "model.joblib")


def generate_features(df, horizon):
    return pd.concat(
        [
            utils.generate_targets(df, horizon, ["was_home"])
            .astype("Int32")
            .astype(float),
            utils.generate_targets(df, horizon, ["total_difficulty"]),
            utils.generate_rolling_features(
                df, ["minutes", "total_points", "xP"], windows=(4,20), aggs=("mean",)
            ),
            utils.generate_lag_features(
                df, ["minutes", "total_points"], lags=(0, 1, 2)
            ),
            utils.generate_lag_features(df, ["value_rank"], lags=(0,)),
        ],
        axis=1,
    )


def get_targets(df, horizon):
    return utils.generate_targets(df, horizon, ["total_points", "minutes", "was_home"])


def train_filter(df, targets):
    return (
        targets["total_points"].notnull()
        & (df["selected_by_percent"] > 1)
        & (df["minutes"] > 0)
    )


def inference_filter(df):
    (
        (df["GW"] == 5)
        & (df["season"] == "2022-23")
    )


def train_test_split(df, features, targets):
    # predicting scores conditioned on player appearing
    train_filter = (
        df["season"].isin(utils.SEASONS[:-2])
    )
    val_filter = (
        df["season"].isin(utils.SEASONS[-2:])
    )
    top_val_filter = (
        df["season"].isin(utils.SEASONS[-2:])
        & (df["selected_by_percent"] > 10)
    )

    train_features = features[train_filter]
    val_features = features[val_filter]
    top_val_features = features[top_val_filter]
    train_targets = targets[train_filter][TARGET_COL]
    val_targets = targets[val_filter][TARGET_COL]
    top_val_targets = targets[top_val_filter][TARGET_COL]
    return train_features, val_features, top_val_features, train_targets, val_targets, top_val_targets


def transform(targets):
    return np.clip(targets, 0, np.inf) ** 1.5


def inverse(targets):
    return np.clip(targets, 0, np.inf) ** 0.66


def train(model, train_features, train_targets, **fit_kwargs):
    model = model.fit(train_features, transform(train_targets), **fit_kwargs)
    return model

def predict(
    model, test_features
):
    return inverse(model.predict(test_features))


def get_scores(preds, targets):
    return {
        "rmse": mean_squared_error(targets, preds) ** 0.5,
        "mae": mean_absolute_error(targets, preds),
    }


class GKModel:
    def generate_features(self, df):
        return pd.concat(
            [
                utils.generate_targets(df, self.horizon, ["was_home"])
                .fillna(True)
                .astype("Int32")
                .astype(float),
                utils.generate_targets(df, self.horizon, ["total_difficulty"]),
                utils.generate_rolling_features(
                    df, ["saves", "minutes"], windows=(4, 19), aggs=("mean",)
                ),
                utils.generate_lag_features(df, ["value_rank"], lags=(0,)),
            ],
            axis=1,
        )


class DFModel:
    def generate_features(self, df):
        return pd.concat(
            [
                utils.generate_targets(df, self.horizon, ["was_home"])
                .fillna(True)
                .astype("Int32")
                .astype(float),
                utils.generate_targets(df, self.horizon, ["total_difficulty"]),
                utils.generate_rolling_features(
                    df, ["xP", "minutes"], windows=(4, 19), aggs=("mean",)
                ),
                utils.generate_lag_features(df, ["value_rank"], lags=(0,)),
            ],
            axis=1,
        )
