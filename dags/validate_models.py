import logging
from collections import defaultdict
import model_utils
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def validate_model(model, model_name):
    pred_list = []
    ys = []
    scores = defaultdict(list)
    for test_week in range(2, 37, 4):
        print(test_week)
        if model_name == "linear":
            Xtrain, Xtest, ytrain, ytest, test_names = model_utils.get_data(test_week=test_week,
                                                                            test_season=2016,
                                                                            one_hot=True)
        else:
            Xtrain, Xtest, ytrain, ytest, test_names = model_utils.get_data(test_week=test_week,
                                                                            test_season=2016,
                                                                            one_hot=False)
        preds = model.fit((Xtrain), ytrain).predict((Xtest))
        imps = None
        if "xgb" in model_name:
            imps = pd.Series(model.booster().get_fscore())
        if imps is not None and test_week == 1:
            logging.info("\n{}".format(imps.sort_values().tail()))
            imps.to_csv("/data/{}_imps.csv".format(model_name))
        pred_list.append(preds)
        ys.append(ytest)
        scores[model_name].append(mean_squared_error(ytest, preds) ** 0.5)

    preds = np.concatenate(pred_list)
    scores = pd.DataFrame(scores, index=range(2, 37, 4))
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
    scores["mean_model"] = [mean_squared_error(y, p) ** 0.5 for y, p in zip(ys, sum_preds)]
    import matplotlib
    matplotlib.use('Agg')
    scores.plot()
    matplotlib.pyplot.savefig("/data/scores.png")
    scores.to_csv("/data/validation_scores.csv")
    logging.info("\n{}".format(scores.mean()))


if __name__ == "__main__":
    import datetime
    validate_models(datetime.datetime.now())
