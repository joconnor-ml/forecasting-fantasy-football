import argparse
import pathlib

import pandas as pd

from fpl_forecast import total_points
from fpl_forecast import utils as forecast_utils


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

    all_scores = []
    for model_name, model in total_points.get_models().items():
        model = total_points.train(
            model, train_features, train_targets
        )
        preds = total_points.predict(model, val_features)
        top_preds = total_points.predict(model, top_val_features)
        scores = total_points.get_scores(val_targets, preds)
        top_scores = total_points.get_scores(top_val_targets, top_preds)
        all_scores.append({"model": model_name, **top_scores})

    all_scores = pd.DataFrame(all_scores).sort_values("rmse")
    best_model_name = all_scores.iloc[0]["model"]
    best_model = total_points.get_models()[best_model_name]
    best_model = total_points.train(
        best_model, pd.concat([train_features, val_features]), pd.concat([train_targets, val_targets])
    )

    test_features = features[total_points.inference_filter(df)]
    final_preds = total_points.predict(best_model, test_features)
    final_preds.to_csv(path=pathlib.Path(output_path) / position / f"{horizon}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--position", type=str, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()
    main(args.position, args.horizon, args.outdir)
