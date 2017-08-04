"""Get data into nice form for training
our models"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV
from sklearn.preprocessing import Imputer, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

import logging


def generate_predictions(execution_date, **kwargs):
    details = pd.read_csv("player_details.csv", index_col=0, nrows=650)

    test_week = 37
    panel = pd.read_pickle("data.pkl").swapaxes(0,2)
    # pick a week to test on:
    test = panel.loc[:, test_week, :]
    ytest = test["target"]
    Xtest = test.drop(["target", "id_mean"], axis=1).astype(np.float64)
    logging.info("\n{}".format(test.head()))

    train = panel.loc[:, 10:test_week, :].to_frame()  # flatten
    ytrain = train["target"]
    Xtrain = train.drop(["target", "id_mean"], axis=1).astype(np.float64)
    
    model = XGBRegressor(n_estimators=32, learning_rate=0.2, max_depth=8)
    model.fit((Xtrain), ytrain)
    preds = model.predict((Xtest))
    notnans = ytest.notnull()
    rmse = mean_squared_error(ytest[notnans], preds[notnans]) ** 0.5
    imps = pd.Series(model.booster().get_fscore())
    logging.info("\n{}".format(imps.sort_values().tail(10)))
    logging.info("")
    preds = pd.DataFrame({"preds": preds, "score":ytest.values}, index=details.web_name)
    logging.info("\n{}".format(preds.sort_values("score").dropna().tail(10)))
    logging.info("RMSE XGB: {}".format(rmse))
    logging.info("")

    model = make_pipeline(Imputer(), #PolynomialFeatures(),
                          MinMaxScaler(), RidgeCV())
    model.fit((Xtrain), ytrain)
    preds = model.predict((Xtest))
    notnans = ytest.notnull()
    rmse = mean_squared_error(ytest[notnans], preds[notnans]) ** 0.5
    imps = pd.Series(model.steps[-1][-1].coef_, index=Xtrain.columns).abs()
    logging.info("\n{}".format(imps.sort_values().tail(10)))
    logging.info("")
    preds = pd.DataFrame({"preds": preds, "score":ytest.values}, index=details.web_name)
    logging.info("\n{}".format(preds.sort_values("score").dropna().tail(10)))
    logging.info("RMSE LR: {}".format(rmse))
    logging.info("")


if __name__ == "__main__":
    build_models()
