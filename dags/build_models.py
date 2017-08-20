import logging
import pickle
import model_utils
import pandas as pd
from pymongo import MongoClient
import os

client = MongoClient(os.environ['MONGO_URL'])
db = client["fantasy_football"]
collection = db["predictions"]


def build_models(execution_date, **kwargs):
    preds = {}
    for name, model in model_utils.models.items():
        logging.info(name)
        if name == "linear":
            Xtrain, Xtest, ytrain, ytest, test_names, test_week = model_utils.get_data(test_week=None,
                                                                                       test_season=2017,
                                                                                       one_hot=True)
        else:
            Xtrain, Xtest, ytrain, ytest, test_names, test_week = model_utils.get_data(test_week=None,
                                                                                       test_season=2017,
                                                                                       one_hot=False)
            Xtest.index = test_names
            Xtest.to_csv("/data/test_features_gw{}.csv".format(test_week))
        model = model.fit(Xtrain, ytrain)
        preds[name] = pd.Series(model.predict(Xtest), index=test_names)
        print(name)
        print(preds[name])
        with open("/models/{}_gw{}.pkl".format(name, test_week), "wb") as f:
            pickle.dump(model, f)
    preds = pd.DataFrame(preds)
    preds["name"] = preds.index
    preds.to_csv("/preds/gw{}.csv".format(test_week), index=False)
    collection.insert_many(preds.to_dict("records"))


if __name__ == "__main__":
    import datetime

    build_models(datetime.datetime.now())
