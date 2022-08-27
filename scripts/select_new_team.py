import pandas as pd
from fpl_opt import selection


def main():
    player_df = pd.read_csv("total_points.csv").rename(columns={"p": "expected_score"})
    player_df = player_df.join(
        pd.read_csv("playing_chance.csv")[["p"]].rename(columns={"p": "playing_chance"})
    )

    decisions, captain_decisions, sub_decisions = selection.select_team(
        player_df["expected_score"],
        player_df["price"],
        player_df["position"].replace({"GKP": "GK"}).map({"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}),
        player_df["team"],
        playing_chance=player_df["playing_chance"],
        sub_factors=[0.15, 0.15, 0.15, 0.05],
    )
    selection_df = selection.get_selection_df(
        decisions, captain_decisions, sub_decisions, player_df
    )
    selection_df[selection.COLS_TO_PRINT].to_csv("selection.csv")


if __name__ == "__main__":
    main()
