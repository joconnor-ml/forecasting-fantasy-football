import argparse
import pathlib

import numpy as np
import pandas as pd

from fpl_forecast import playing_chance
from fpl_forecast import utils as forecast_utils


def main(position: str, horizon: int):
    df = forecast_utils.get_player_data(seasons=forecast_utils.SEASONS)

    all_scores = []
    for model_name, model in playing_chance.get_models(position, horizon).items():
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

    all_scores = pd.DataFrame(all_scores).sort_values("log_loss")
    best_model_name = all_scores.iloc[-1]["model"]
    best_model = playing_chance.get_models(position, horizon)[best_model_name]
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
        ["name", "team", "position", "value", "value_rank", "minutes", "total_points"],
    ]
    out_df["playing_chance"] = best_model.predict(test_features)
    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--position", type=str, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()
    out_df = main(args.position, args.horizon)
    output_path = (
        pathlib.Path(args.outdir) / args.position / f"playing_chance_{args.horizon}.csv"
    )
    output_path.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path)
