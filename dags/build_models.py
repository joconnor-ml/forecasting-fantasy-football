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
            Xtrain, Xtest, ytrain, ytest, dftest, test_week = model_utils.get_data(test_week=39,
                                                                                   test_season=2017,
                                                                                   one_hot=True)
        else:
            Xtrain, Xtest, ytrain, ytest, dftest, test_week = model_utils.get_data(test_week=39,
                                                                                   test_season=2017,
                                                                                   one_hot=False)
        model = model.fit(Xtrain, ytrain)
        preds[name] = pd.Series(model.predict(Xtest)).values
        with open("/models/{}_gw{}.pkl".format(name, test_week), "wb") as f:
            pickle.dump(model, f)

    teams = []
    for team in db["teams"].find({"season": 2017}):
        team["next_opponent"] = team["next_event_fixture"][0]["opponent"]
        team["is_home"] = team["next_event_fixture"][0]["is_home"]
        teams.append(team)
    team_df = pd.DataFrame(teams)

    dftest = pd.merge(dftest, team_df[["name", "is_home", "next_opponent", "code"]], left_on="team_code",
                      right_on="code")
    dftest = pd.merge(dftest, team_df[["name", "code"]], left_on="next_opponent", right_on="code").reset_index()
    preds = pd.DataFrame(preds)
    print(preds.head())
    preds = pd.concat([preds, dftest], axis=1)

    preds.to_csv("/preds/gw{}.csv".format(test_week))
    collection.drop()
    collection.insert_many(preds.to_dict("records"))


if __name__ == "__main__":
    import datetime

    build_models(datetime.datetime.now())
