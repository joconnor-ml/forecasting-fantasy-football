import argparse

import pandas as pd

from fpl_opt import selection


def main(points_forecast, playing_forecast, budget):
    player_df = pd.read_parquet(points_forecast)
    player_df = player_df.merge(
        pd.read_parquet(playing_forecast)[["playing_chance", "element", "horizon"]],
        on=["element", "horizon"]
    )
    player_df = player_df.groupby("element").agg({
        "score_pred": "mean", "playing_chance": "mean", "value": "first", "position": "first", "team": "first",
        "name": "first"
    })
    decisions, captain_decisions, sub_decisions = selection.select_team(
        player_df["score_pred"].values,
        player_df["value"].values,
        player_df["position"]
        .replace({"GKP": "GK"})
        .map({"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}).values,
        player_df["team"].values,
        playing_chance=player_df["playing_chance"].values,
        sub_factors=[0.15, 0.15, 0.15, 0.05],
        total_budget=budget
    )
    selection_df = selection.get_selection_df(
        decisions, captain_decisions, sub_decisions, player_df
    )
    selection_df["name"] = player_df.iloc[selection_df.index]["name"].values
    selection_df["value"] = player_df.iloc[selection_df.index]["value"].values
    selection_df["position"] = player_df.iloc[selection_df.index]["position"].values
    selection_df["team"] = player_df.iloc[selection_df.index]["team"].values
    selection_df["playing_chance"] = player_df.iloc[selection_df.index]["playing_chance"].values
    selection_df["score_pred"] = player_df.iloc[selection_df.index]["score_pred"].values
    return selection_df#[selection.COLS_TO_PRINT]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--points_forecast", type=str, default="prod/points.pq")
    parser.add_argument("--playing_forecast", type=str, default="prod/playing.pq")
    parser.add_argument("--budget", type=float, default=1000)
    parser.add_argument("--output_parquet", type=str, default="team_selection.pq")
    args = parser.parse_args()

    df = main(points_forecast=args.points_forecast, playing_forecast=args.playing_forecast, budget=args.budget)
    print(df.sort_values("position"))
    print(df.sum())
    df.to_parquet(args.output_parquet)