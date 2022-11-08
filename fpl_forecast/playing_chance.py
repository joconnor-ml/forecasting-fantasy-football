import json
import pathlib

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from . import utils

TARGET_COL = "played"


def get_models(position, horizon):
    model_class = PlayingChanceModel
    return {
        **{
            f"lasso_{alpha}": model_class(
                model=make_pipeline(
                    SimpleImputer(), StandardScaler(), LogisticRegression(C=alpha)
                ),
                horizon=horizon,
            )
            for alpha in [0.1, 1, 10, 100]
        },
        **{
            f"lasso_poly_{alpha}": model_class(
                model=make_pipeline(
                    SimpleImputer(),
                    PolynomialFeatures(),
                    StandardScaler(),
                    LogisticRegression(C=alpha),
                ),
                horizon=horizon,
            )
            for alpha in [0.1, 1, 10]
        },
    }


class PlayingChanceModel:
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
            df, self.horizon, ["played"]
        )

    def train_filter(self, df, targets):
        return targets["played"].notnull() & (df["selected_by_percent"] > 1)

    def inference_filter(self, df, targets):
        # TODO automate getting the inference week
        return df["total_points"].notnull() & (df["season"] == utils.SEASONS[-1]) & (df["GW"] == df[(df["season"] == utils.SEASONS[-1]) & df["total_points"].notnull()]["GW"].max())
    
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

    def train(self, train_features, train_targets, **fit_kwargs):
        self.model = self.model.fit(
            train_features, train_targets, **fit_kwargs
        )
        return self

    def predict(self, test_features):
        return self.model.predict_proba(test_features)[:, -1]

    def get_scores(self, preds, targets):
        return {
            "log_loss": log_loss(targets, preds),
            "roc_auc": roc_auc_score(targets, preds),
            "accuracy": accuracy_score(targets, preds > 0.5),
        }

    def generate_features(self, df):
        return pd.concat(
            [
                utils.generate_rolling_features(
                    df, ["minutes", "total_points", "xP"], windows=(3, 10), aggs=("mean",)
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
