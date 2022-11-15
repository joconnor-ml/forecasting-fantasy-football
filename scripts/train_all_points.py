import argparse

import pandas as pd

from . import train_points_model


def main(max_horizon, output_path, scores_path):
    all_preds = []
    all_scores = []
    for position in ["GK", "DEF", "MID", "FWD"]:
        for horizon in range(1, max_horizon + 1):
            pred_df, score_df = train_points_model.main(position, horizon)
            pred_df["horizon"] = horizon
            score_df["horizon"] = horizon
            score_df["position"] = position
            all_preds.append(pred_df)
            all_scores.append(score_df)
    pd.concat(all_preds).to_parquet(output_path)
    pd.concat(all_scores).to_parquet(scores_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--outfile", type=str, required=True)
    parser.add_argument("--score_path", type=str, required=True)
    args = parser.parse_args()
    main(args.horizon, args.outfile, args.score_path)
