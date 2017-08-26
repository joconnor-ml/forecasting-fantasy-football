import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer, MinMaxScaler, PolynomialFeatures, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor
from bayesian_models import BayesianPointsRegressor, MeanPointsRegressor


def get_data(test_week, test_season, one_hot):
    df = pd.read_csv("/data/data.csv", index_col=0).reset_index()
    df = df[~(df["target_minutes"] < 60)]  # keep nans and >=60
    if one_hot:
        # opponent_team = pd.get_dummies(df["target_team"].fillna(999).astype(int)).add_prefix("opponent_")
        # own_team = pd.get_dummies(df["team_code"].fillna(999).astype(int)).add_prefix("team_")
        position = pd.get_dummies(df["element_type"].fillna(999).astype(int)).add_prefix("position_")
        df = pd.concat([
            df,
            # opponent_team,
            # own_team,
            position,
        ], axis=1)
        X = df.drop(["target", "id", "target_minutes", "value",
                     "web_name", "index", "season",
                     "gameweek", "target_team", "team_code", "element_type"], axis=1).astype(np.float64)
    else:
        X = df.drop(["target", "id", "target_minutes", "value",
                     "web_name", "index", "team_code",
                     "season", "gameweek"], axis=1).astype(np.float64)
    X = X.dropna(how="all", axis=1)
    try:
        X = X.loc[:, ((X != X.iloc[0, :]) &
                      (X.notnull())).any()]
        print(X.shape)
    except:
        print("Error")
        print(X.iloc[0, :])
    y = df["target"]

    info_features = ["web_name", "team_code", "gameweek", "element_type", "value"]
    if test_week is not None:
        notnull = y.notnull()
        X = X[notnull]
        y = y[notnull]
        df = df[notnull]
        train = df["gameweek"] < test_week
        test = df["gameweek"] == test_week
        return X.loc[train], X.loc[test], y.loc[train], y.loc[test], df.loc[test, info_features]
    else:
        test = (df["season"] == 2017) & (df["target"].isnull())
        train = y.notnull()
        return X.loc[train], X.loc[test], y.loc[train], y.loc[test], df.loc[test, info_features]

models = {
    "xgb":
    XGBRegressor(n_estimators=64, learning_rate=0.1, max_depth=1),
    "xgb2":
    XGBRegressor(n_estimators=64, learning_rate=0.1, max_depth=2),
    "xgb3":
    XGBRegressor(n_estimators=128, learning_rate=0.1, max_depth=2),
    "rf":
    make_pipeline(Imputer(), RandomForestRegressor(n_estimators=100, max_depth=3)),
    "linear":
    make_pipeline(
        Imputer(), MinMaxScaler(), RidgeCV(),
    ),
    "constant":
        make_pipeline(
            Imputer(), MinMaxScaler(), Ridge(1e-9),
        ),
    "linear1":
        make_pipeline(
            Imputer(), MinMaxScaler(), Ridge(1),
        ),
    "linear2":
        make_pipeline(
            Imputer(), MinMaxScaler(), Ridge(100),
        ),
    "linear3":
        make_pipeline(
            Imputer(), MinMaxScaler(), Ridge(1000),
        ),
    # "polynomial_pca":
    # make_pipeline(
    #    Imputer(), PolynomialFeatures(), MinMaxScaler(), RidgeCV(),
    # ),
    "polynomial_fs":
    make_pipeline(
        Imputer(), PolynomialFeatures(), MinMaxScaler(), SelectKBest(f_regression, k=32),  RidgeCV(),
    ),
    "simple_mean":
    MeanPointsRegressor(),
    "bayes_global_prior":
    BayesianPointsRegressor("global"),
    # "bayes_team_prior":
    # BayesianPointsRegressor("team"),
    "bayes_position_prior":
    BayesianPointsRegressor("position"),
}
