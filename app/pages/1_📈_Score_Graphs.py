import plotly.express as px
import streamlit as st

import utils


def main():
    settings = utils.get_settings()

    st.set_page_config(page_title="Score Graphs", page_icon="📈")

    st.markdown("# Predicted Score Graphs")
    st.sidebar.header("Predicted Score Graphs")

    df = utils.get_forecast_data(settings.points_data_path, settings.playing_data_path, settings.bucket_name)
    players = st.multiselect("Choose players", list(df.index.unique()))
    if not players:
        st.error("Please select at least one player.")
    else:
        fig = px.line(
            df.loc[players].reset_index(),
            x="horizon",
            y="score_if_playing",
            color="name",
            hover_data=["name", "score_if_playing", "playing_chance"],
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
