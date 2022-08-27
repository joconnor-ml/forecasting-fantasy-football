import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from xgboost import XGBRegressor

from . import utils

TARGET_COL = "total_points"


def get_models():
    return [
        make_pipeline(
            SimpleImputer(), PolynomialFeatures(), StandardScaler(), LinearRegression()
        ),
        XGBRegressor(),
    ]


def generate_features(df):
    return pd.concat(
        [
            sum(utils.generate_targets(df, i, ["was_home"]) for i in [1, 2, 3, 4, 5])
            .astype("Int32")
            .astype(float),
            utils.generate_rolling_features(
                df, ["minutes", "total_points", "xP"], windows=(3,), aggs=("mean",)
            ),
            utils.generate_lag_features(
                df, ["minutes", "total_points"], lags=(0, 1, 2)
            ),
            utils.generate_lag_features(
                df, ["DEF", "FWD", "GK", "MID", "value_rank"], lags=(0,)
            ),
        ],
        axis=1,
    )


def get_targets(df):
    return sum(
        utils.generate_targets(df, i, ["total_points", "minutes", "was_home"])
        for i in [1, 2, 3, 4, 5]
    )


def train_test_split(df, features, targets):
    test_teams = ["Crystal Palace", "Newcastle", "Brighton", "Wolves", "Liverpool"]

    # predicting scores conditioned on player appearing
    train_filter = (
        (targets["minutes"] > 0)
        & ~(df["team"].isin(test_teams))
        & targets[TARGET_COL].notnull()
    )
    val_filter = (
        (targets["minutes"] > 0)
        & df["team"].isin(test_teams)
        & targets[TARGET_COL].notnull()
    )

    train_features = features[train_filter]
    val_features = features[val_filter]
    train_targets = targets[train_filter][TARGET_COL]
    val_targets = targets[val_filter][TARGET_COL]
    return train_features, val_features, train_targets, val_targets


def test_model(
    model, train_features, train_targets, test_features, test_targets, **fit_kwargs
):
    model = model.fit(train_features, train_targets, **fit_kwargs)
    preds = model.predict(test_features)
    get_scores(preds, test_targets)
    return model, preds


def get_scores(preds, targets):
    print(
        mean_squared_error(targets, preds) ** 0.5, mean_absolute_error(targets, preds)
    )

def predict(model, features):
    return model.predict(features)