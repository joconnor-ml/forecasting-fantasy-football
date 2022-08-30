import argparse
import pathlib

import pandas as pd

from fpl_forecast import total_points, playing_chance
from fpl_forecast import utils as forecast_utils

TASKS = {"total_points": total_points, "playing_chance": playing_chance}


def main(position: str, horizon: int, output_path: str):
    df = forecast_utils.get_player_data(seasons=forecast_utils.SEASONS)
    df = df[df["position"] == position]

    targets = total_points.get_targets(df, horizon=horizon)
    features = total_points.generate_features(df, horizon=horizon)

    train_filter = total_points.train_filter(df, targets)
    df = df[train_filter]
    targets = targets[train_filter]
    features = features[train_filter]

    (
        train_features,
        val_features,
        top_val_features,
        train_targets,
        val_targets,
        top_val_targets,
    ) = total_points.train_test_split(df, features, targets)

    ## benchmark:
    benchmark_pred = pd.np.ones_like(val_targets) * val_targets.mean()
    total_points.get_scores(benchmark_pred, val_targets)

    for model_name, model in total_points.get_models().items():
        model, preds, top_preds = total_points.test_model(
            model, train_features, train_targets, val_features, top_val_features
        )
        scores = total_points.get_scores(val_targets, preds)
        top_scores = total_points.get_scores(top_val_targets, top_preds)
        json = {"scores": scores, "top_scores": top_scores}
        total_points.save_model(
            model, json, path=pathlib.Path(output_path) / position / str(horizon) / model_name
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--position", type=str, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()
    main(args.position, args.horizon, args.outdir)
