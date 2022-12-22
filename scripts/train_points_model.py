import numpy as np
import pandas as pd

from fpl_forecast import total_points
from fpl_forecast import utils as forecast_utils


def main(position: str, horizon: int):
    df = forecast_utils.get_player_data(seasons=forecast_utils.SEASONS)
    df = df[
        ((df["position"] == position) & (df["minutes"] > 0))
        | (
            (df["season"] == forecast_utils.SEASONS[-1])
            & (df["position"] == position)
            & (
                df["GW"]
                == df[
                    (df["season"] == forecast_utils.SEASONS[-1])
                    & df["total_points"].notnull()
                ]["GW"].max()
            )
        )
    ]

    all_scores = []
    for model_name, model in total_points.get_models(position, horizon).items():
        targets = model.get_targets(df)
        features = model.generate_features(df)
        train_filter = model.train_filter(df, targets)

        (
            train_features,
            val_features,
            top_val_features,
            train_targets,
            val_targets,
            top_val_targets,
        ) = model.train_test_split(
            df[train_filter], features[train_filter], targets[train_filter]
        )

        ## benchmark:
        benchmark_pred = np.ones_like(val_targets) * val_targets.mean()
        benchmark_scores = model.get_scores(val_targets, benchmark_pred)
        model = model.train(train_features, train_targets)
        preds = model.predict(val_features)
        top_preds = model.predict(top_val_features)
        scores = model.get_scores(val_targets, preds)
        top_scores = model.get_scores(top_val_targets, top_preds)
        all_scores.append({"model": model_name, **top_scores})

    all_scores = pd.DataFrame(all_scores).sort_values("rmse")
    best_model_name = all_scores.iloc[0]["model"]
    best_model = total_points.get_models(position, horizon)[best_model_name]
    targets = best_model.get_targets(df)
    features = best_model.generate_features(df)

    train_filter = best_model.train_filter(df, targets)

    (
        train_features,
        val_features,
        top_val_features,
        train_targets,
        val_targets,
        top_val_targets,
    ) = best_model.train_test_split(
        df[train_filter], features[train_filter], targets[train_filter]
    )

    best_model = best_model.train(
        pd.concat([train_features, val_features]),
        pd.concat([train_targets, val_targets]),
    )

    inference_filter = best_model.inference_filter(df, targets)
    test_features = features[inference_filter]
    out_df = df.loc[
        inference_filter,
        [
            "name",
            "element",
            "team",
            "position",
            "value",
            "value_rank",
            "minutes",
            "total_points",
            "opponent",
        ],
    ]
    out_df["score_pred"] = best_model.predict(test_features)
    return out_df, all_scores, test_features
