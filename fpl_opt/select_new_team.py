import argparse

import pandas as pd

from fpl_opt import selection


def main(budget):
    player_df = pd.read_parquet("total_points.pq").rename(
        columns={"p": "expected_score"}
    )
    player_df = player_df.join(
        pd.read_parquet("playing_chance.pq")[["p"]].rename(
            columns={"p": "playing_chance"}
        )
    )

    decisions, captain_decisions, sub_decisions = selection.select_team(
        player_df["expected_score"],
        player_df["price"],
        player_df["position"]
        .replace({"GKP": "GK"})
        .map({"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}),
        player_df["team"],
        playing_chance=player_df["playing_chance"],
        sub_factors=[0.15, 0.15, 0.15, 0.05],
        total_budget=budget
    )
    selection_df = selection.get_selection_df(
        decisions, captain_decisions, sub_decisions, player_df
    )
    return selection_df[selection.COLS_TO_PRINT]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=100)
    parser.add_argument("--output_parquet", type=str, default=None)
    args = parser.parse_args()

    df = main(budget=args.budget)
    df.to_parquet(args.output_parquet)