import logging
from collections import defaultdict
import model_utils
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pymongo import MongoClient  # Database connector
import os

client = MongoClient(os.environ['MONGO_URL'])
db = client["fantasy_football"]  # Select the database
score_db = db["scores"]  # Select the collection

test_weeks = list(range(2, 37)) + [39, 40, 41]


def validate_model(model, model_name):
    pred_list = []
    ys = []
    scores = defaultdict(list)
    mae_scores = defaultdict(list)
    for test_week in test_weeks:
        if model_name == "linear":
            Xtrain, Xtest, ytrain, ytest, df_train, df_test = model_utils.get_data(test_week=test_week,
                                                                                   test_season=2016,
                                                                                   one_hot=True)
        else:
            Xtrain, Xtest, ytrain, ytest, df_train, df_test = model_utils.get_data(test_week=test_week,
                                                                                   test_season=2016,
                                                                                   one_hot=False)
        preds = model.fit(Xtrain, ytrain).predict((Xtest))
        imps = None
        if "xgb" in model_name:
            imps = pd.Series(model.booster().get_fscore()).sort_values(ascending=False)
        # if "linear" in model_name:
        #    try:
        #        imps = pd.Series(model.steps[-1][-1].coef_, index=Xtrain.columns).sort_values(ascending=False)
        #    except Exception as e:
        #        logging.warning("Saving importances failed:")
        #        logging.warning(e)
        if imps is not None and (test_week == 37 or test_week == 39):
            logging.info("\n{}".format(imps.head()))
            imps.to_csv("/data/{}_imps_{}.csv".format(model_name, test_week))
        pred_list.append(preds)
        ys.append(ytest)
        scores[model_name].append(mean_squared_error(ytest, preds) ** 0.5)
        mae_scores[model_name].append(mean_absolute_error(ytest, preds))

    preds = np.array(pred_list)
    scores = pd.DataFrame(scores, index=test_weeks)
    mae_scores = pd.DataFrame(mae_scores, index=test_weeks)
    return ys, preds, scores, mae_scores


def validate_model_season2(model, model_name):
    pred_list = []
    ys = []
    scores = defaultdict(list)
    test_week = 39
    train_week = 1
    if model_name == "linear":
        Xtrain, Xtest, ytrain, ytest, df_train, df_test = model_utils.get_data(test_week=test_week,
                                                                               test_season=2016,
                                                                               one_hot=True)
    else:
        Xtrain, Xtest, ytrain, ytest, df_train, df_test = model_utils.get_data(test_week=test_week,
                                                                               test_season=2016,
                                                                               one_hot=False)

    preds = model.fit(Xtrain.loc[df_train.gameweek == train_week],
                      ytrain.loc[df_train.gameweek == train_week]).predict(Xtest)
    imps = None
    if "xgb" in model_name:
        imps = pd.Series(model.booster().get_fscore()).sort_values(ascending=False)
    if "linear" in model_name:
        try:
            imps = pd.Series(model.steps[-1][-1].coef_, index=Xtrain.columns).sort_values(ascending=False)
        except Exception as e:
            logging.warning("Saving importances failed:")
            logging.warning(e)
    if imps is not None:
        logging.info("\n{}".format(imps.head()))
        imps.to_csv("/data/{}_imps_{}.csv".format(model_name, test_week))
    return mean_squared_error(ytest, preds)


def validate_models(execution_date, **kwargs):
    sum_preds = None
    all_scores = []
    all_mae_scores = []
    for name, model in model_utils.models.items():
        logging.info(name)
        ys, preds, scores, mae_scores = validate_model(model, name)
        preds = np.array(preds)
        if name in ["xgb", "grouped", "linear2", "rf", "polynomial_fs"]:
            if sum_preds is None:
                sum_preds = preds
            else:
                sum_preds += preds
        all_scores.append(scores)
        all_mae_scores.append(mae_scores)
    sum_preds = sum_preds / 4
    scores = pd.concat(all_scores, axis=1)
    scores["mean_model"] = [mean_squared_error(y, p) ** 0.5 for y, p in zip(ys, sum_preds)]
    scores.to_csv("/data/validation_scores.csv")

    mae_scores = pd.concat(all_mae_scores, axis=1)
    mae_scores["mean_model"] = [mean_absolute_error(y, p) for y, p in zip(ys, sum_preds)]
    mae_scores.to_csv("/data/validation_mae_scores.csv")

    score_db.drop()
    score_db.insert_many(scores.reset_index().to_dict("records"))
    logging.info("\n{}".format(scores.mean()))
    logging.info("\n{}".format(mae_scores.mean()))

    for name, model in model_utils.models.items():
        logging.info(name)
        score = validate_model_season2(model, name)
        logging.info(score)

if __name__ == "__main__":
    import datetime
    validate_models(datetime.datetime.now())
