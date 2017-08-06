"""Get data into nice form for training
our models"""

import logging
import pickle

import model_utils


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
        with open("/forecasting-fantasy-football/models/{}_gw{}.pkl".format(name, test_week), "wb") as f:
            pickle.dump(model, f)
        preds.to_csv("/forecasting-fantasy-football/preds/{}_gw{}.csv".format(name, test_week))


if __name__ == "__main__":
    import datetime
    build_models(datetime.datetime.now())
