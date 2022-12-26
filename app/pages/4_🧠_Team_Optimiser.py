import streamlit as st

import fpl_opt.selection
import utils


def pick_team(points_data_path, playing_data_path, bucket_name, budget):
    player_df = utils.get_forecast_data(
        points_data_path, playing_data_path, bucket_name
    ).reset_index()
    player_df = player_df.groupby("element").agg(
        {
            "score_pred": "mean",
            "playing_chance": "mean",
            "value": "first",
            "position": "first",
            "team": "first",
            "name": "first",
        }
    )
    decisions, captain_decisions, sub_decisions = fpl_opt.selection.select_team(
        player_df["score_pred"].values,
        player_df["value"].values,
        player_df["position"]
        .replace({"GKP": "GK"})
        .map({"GK": 1, "DEF": 2, "MID": 3, "FWD": 4})
        .values,
        player_df["team"].values,
        playing_chance=player_df["playing_chance"].values,
        sub_factors=[0.15, 0.15, 0.15, 0.05],
        total_budget=budget,
    )
    selection_df = fpl_opt.selection.get_selection_df(
        decisions, captain_decisions, sub_decisions, player_df
    )

    return selection_df


settings = utils.get_settings()


def main():
    budget = float(st.text_input("Budget", "100"))
    selection_df = utils.pick_team(
        settings.points_data_path,
        settings.playing_data_path,
        settings.bucket_name,
        budget,
    )
    st.dataframe(
        selection_df,
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
