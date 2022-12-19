from io import StringIO

import pandas as pd
import plotly.express as px
import streamlit as st

BUCKET_NAME = "forecasting-fantasy-football"
POINTS_DATA_PATH = "prod/points.pq"
PLAYING_DATA_PATH = "prod/playing.pq"
MAX_HORIZONS = 5


def read_gcs_file(path):
    from google.cloud.storage import Client

    client = Client()
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.get_blob(path)
    return StringIO(blob.download_as_text(encoding="utf-8"))


@st.cache
def get_points_data(path):
    df = pd.read_parquet(read_gcs_file(path), index_col=0)
    return df


@st.cache
def get_playing_data(path):
    df = pd.read_parquet(read_gcs_file(path), index_col=0)
    return df


@st.cache
def get_forecast_data(points_path, playing_path):
    playing = get_playing_data(playing_path)
    points = get_points_data(points_path)[["name", "team", "score_pred", "horizon"]]
    df = playing.merge(points, how="left", on=["name", "team", "horizon"]).set_index(
        "name"
    )[["horizon", "score_pred", "playing_chance"]]
    df = df.rename(columns={"score_pred": "score_if_playing"})
    df["score_pred"] = df["score_if_playing"] * df["playing_chance"]
    return df


def main():
    df = get_forecast_data(POINTS_DATA_PATH, PLAYING_DATA_PATH)
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
