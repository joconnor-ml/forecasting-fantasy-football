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
    for name, model in model_utils.models.items():
        logging.info(name)
        if name == "linear":
            Xtrain, Xtest, ytrain, ytest, test_names, test_week = model_utils.get_data(test_week=None,
                                                                            one_hot=True)
        else:
            Xtrain, Xtest, ytrain, ytest, test_names, test_week = model_utils.get_data(test_week=None,
                                                                one_hot=False)
            Xtest.index = test_names
            Xtest.to_csv("test_features_gw{}.csv".format(test_week))
        model = model.fit(Xtrain, ytrain)
        preds = pd.DataFrame({"prediction": model.predict(Xtest)}, index=test_names)
        with open("models/{}_gw{}.pkl".format(name, test_week), "wb") as f:
            pickle.dump(model, f)
        preds.to_csv("preds/{}_gw{}.csv".format(name, test_week))


if __name__ == "__main__":
    import datetime
    build_models(datetime.datetime.now())
