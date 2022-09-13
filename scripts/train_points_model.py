import argparse
import pathlib

import pandas as pd

from fpl_forecast import total_points
from fpl_forecast import utils as forecast_utils


def main(position: str, horizon: int, output_path: str):
    df = forecast_utils.get_player_data(seasons=forecast_utils.SEASONS)
    df = df[(df["position"] == position) & df["minutes"] > 0]

    all_scores = []
    for model_name, model in total_points.get_models(position, horizon).items():
        train_filter = model.train_filter(df, targets)
        targets = model.get_targets(df)
        features = model.generate_features(df)

        targets = targets[train_filter]
        features = features[train_filter]

        (
            train_features,
            val_features,
            top_val_features,
            train_targets,
            val_targets,
            top_val_targets,
        ) = model.train_test_split(df, features, targets)

        ## benchmark:
        benchmark_pred = pd.np.ones_like(val_targets) * val_targets.mean()
        model.get_scores(benchmark_pred, val_targets)
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
    ) = best_model.train_test_split(df, features, targets)

    best_model = best_model.train(
        pd.concat([train_features, val_features]),
        pd.concat([train_targets, val_targets]),
    )

    test_features = features[best_model.inference_filter(df)]
    final_preds = best_model.predict(test_features)
    final_preds.to_csv(path=pathlib.Path(output_path) / position / f"{horizon}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--position", type=str, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()
    main(args.position, args.horizon, args.outdir)
