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
import model_utils

import logging


def validate_model(model, model_name):
    pred_list = []
    ys = []
    scores = defaultdict(list)
    for test_week in range(12, 37, 4):
        if model_name == "linear":
            Xtrain, Xtest, ytrain, ytest, test_names, _ = model_utils.get_data(test_week=test_week,
                                                                one_hot=True)
        else:
            Xtrain, Xtest, ytrain, ytest, test_names, _ = model_utils.get_data(test_week=test_week,
                                                                one_hot=False)
            
        preds = model.fit((Xtrain), ytrain).predict((Xtest))
        rmse = mean_squared_error(ytest, preds) ** 0.5
        imps = None
        if "xgb" in model_name:
            imps = pd.Series(model.booster().get_fscore())
        elif model_name == "linear":
            imps = pd.Series(model.steps[-1][-1].coef_,
                             index=Xtrain.columns)
        if imps is not None and test_week==36:
            logging.info("\n{}".format(imps.sort_values().tail()))
            imps.to_csv("{}_imps.csv".format(model_name))
        pred_list.append(preds)
        ys.append(ytest)
        scores[model_name].append(mean_squared_error(ytest, preds) ** 0.5)

    ytest = np.concatenate(ys)
    preds = np.concatenate(pred_list)
    rmse = mean_squared_error(ytest, preds) ** 0.5
    scores = pd.DataFrame(scores, index=range(12, 37, 4))
    return ys, preds, scores

    
def validate_models(execution_date, **kwargs):
    sum_preds = None
    all_scores = []
    for name, model in model_utils.models.items():
        logging.info(name)
        ys, preds, scores = validate_model(model, name)
        preds = np.array(preds)
        if sum_preds is None:
            sum_preds = preds
        else:
            sum_preds += preds
        all_scores.append(scores)
    sum_preds = sum_preds / len(model_utils.models)
    print(sum_preds)
    print(sum_preds.shape)
    scores = pd.concat(all_scores, axis=1)
    import matplotlib
    matplotlib.use('Agg')
    import seaborn as sns
    #matplotlib.pyplot.plot(scores)
    #matplotlib.pyplot.legend(loc="best")
    scores.plot()
    matplotlib.pyplot.savefig("scores.png")
    logging.info("\n{}".format(scores.mean()))

    #for y, p in zip(ys, preds):
    #    print(mean_squared_error(y, p)**0.5)

if __name__ == "__main__":
    import datetime
    validate_models(datetime.datetime.now())
