import plotly.express as px
import streamlit as st

import utils

settings = utils.get_settings()


def main():
    utils.setup_page("Team Analysis", icon="📊")
    st.markdown("### Import your team:")
    st.markdown(
        "Copy your team code here from the `points` page on FPL, "
        "e.g. for my team: [https://fantasy.premierleague.com/entry/"
        "**7148592**/event/16](https://fantasy.premierleague.com/entry/7148592/event/16)"
    )
    team_id = int(st.text_input("Team ID", "7148592"))
    team_data = utils.get_team_data(team_id, gameweek=utils.get_current_gameweek())
    team_data = team_data.rename(
        columns={"web_name": "Name", "now_cost": "Price", "event_points": "Points"}
    ).set_index("Name")
    team_data["Price"] /= 10
    st.dataframe(
        team_data.loc[team_data["position"] <= 11, ["Price", "Points"]],
        use_container_width=True,
    )
    st.markdown("Subs")
    st.dataframe(
        team_data.loc[team_data["position"] > 11, ["Price", "Points"]],
        use_container_width=True,
    )

    st.markdown("### Team analysis:")
    st.markdown("Your team is very good, well done.")

    preds = utils.get_forecast_data(
        settings.points_data_path, settings.playing_data_path, settings.bucket_name
    ).reset_index()
    team_data_with_preds = team_data.merge(
        preds[["element", "score_if_playing", "playing_chance", "horizon", "name"]],
        left_on="id",
        right_on="element",
        how="left",
    )
    team_data_with_preds["total_expected_score"] = (
        team_data_with_preds["score_if_playing"]
        * team_data_with_preds["playing_chance"]
    )
    scores_by_week = (
        team_data_with_preds.query("position<=11")
        .groupby("horizon")[["total_expected_score"]]
        .sum()
        + 0.1
        * team_data_with_preds.query("position>11")
        .groupby("horizon")[["total_expected_score"]]
        .sum()
    )
    fig = px.line(
        scores_by_week.reset_index(),
        x="horizon",
        y="total_expected_score",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        team_data_with_preds[["name", "score_if_playing", "playing_chance", "horizon"]],
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
