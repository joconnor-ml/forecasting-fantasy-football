import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from xgboost import XGBClassifier

from . import utils

TARGET_COL = "played"


def get_models():
    return [
        make_pipeline(
            SimpleImputer(),
            PolynomialFeatures(),
            StandardScaler(),
            LogisticRegression(),
        ),
        make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression()),
        XGBClassifier(),
    ]


def generate_features(df):
    return pd.concat(
        [
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
    return utils.generate_targets(df, 1, ["played"])


def train_test_split(df, features, targets):
    test_teams = ["Crystal Palace", "Newcastle", "Brighton", "Wolves", "Liverpool"]

    # predicting scores conditioned on player appearing
    train_filter = ~(df["team"].isin(test_teams)) & targets[TARGET_COL].notnull()
    val_filter = df["team"].isin(test_teams) & targets[TARGET_COL].notnull()

    train_features = features[train_filter]
    val_features = features[val_filter]
    train_targets = targets[train_filter][TARGET_COL]
    val_targets = targets[val_filter][TARGET_COL]
    return train_features, val_features, train_targets, val_targets


def test_model(
    model, train_features, train_targets, test_features, test_targets, **fit_kwargs
):
    model = model.fit(train_features, train_targets, **fit_kwargs)
    preds = model.predict_proba(test_features)[:, -1]
    get_scores(preds, test_targets)
    return model, preds


def get_scores(preds, targets):
    print(
        log_loss(targets, preds),
        roc_auc_score(targets, preds),
        accuracy_score(targets, preds > 0.5),
    )


def predict(model, features):
    return model.predict_proba(features)[:, 1]
