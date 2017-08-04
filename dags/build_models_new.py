"""Get data into nice form for training
our models"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV
from sklearn.preprocessing import Imputer, MinMaxScaler, PolynomialFeatures, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import make_pipeline, make_union
from xgboost import XGBRegressor
from sklearn_pandas import DataFrameMapper
from collections import defaultdict
import pickle
import model_utils

import logging


def build_models(execution_date, **kwargs):
    test_week = 37
    for name, model in model_utils.models.items():
        logging.info(name)
        if model_name == "linear":
            Xtrain, Xtest, ytrain, ytest = model_utils.get_data(test_week=test_week,
                                                                one_hot=True)
        else:
            Xtrain, Xtest, ytrain, ytest = model_utils.get_data(test_week=test_week,
                                                                one_hot=False)
        model = model.fit(Xtrain, ytrain)
        preds = model.predict(Xtest)
        with open("models/{}_gw{}.pkl".format(name, test_week), "wb") as f:
            pickle.dump(model, f)
        preds.to_csv("preds/{}_gw{}.csv".format(name, test_week))


if __name__ == "__main__":
    import datetime
    build_models(datetime.datetime.now())
