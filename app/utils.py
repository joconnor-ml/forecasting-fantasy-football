import io
from functools import lru_cache

import pandas as pd
import requests
import streamlit as st
from google.cloud.storage import Client
from pydantic import BaseSettings


class Settings(BaseSettings):
    bucket_name: str = "forecasting-fantasy-football"
    points_data_path: str = "prod/points.pq"
    playing_data_path: str = "prod/playing.pq"
    points_models_data: str = "prod/points_scores.pq"
    playing_models_data: str = "prod/playing_scores.pq"
    forecast_horizon: int = 5

    class Config:
        env_file = ".env"


@lru_cache
def get_settings():
    return Settings()


@st.cache
def read_gcs_file(path, bucket_name):
    client = Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.get_blob(path)
    buffer = io.BytesIO()
    blob.download_to_file(buffer)
    return buffer


@st.cache
def read_parquet_cached(path, bucket_name=None):
    if bucket_name:
        return pd.read_parquet(read_gcs_file(path, bucket_name))
    else:
        return pd.read_parquet(path)


@st.cache
def get_forecast_data(points_path, playing_path, bucket_name=None):
    playing = read_parquet_cached(playing_path, bucket_name)
    points = read_parquet_cached(points_path, bucket_name)[
        ["name", "team", "score_pred", "horizon"]
    ]
    df = playing.merge(points, how="left", on=["name", "team", "horizon"]).set_index(
        "name"
    )[["horizon", "score_pred", "playing_chance"]]
    df = df.rename(columns={"score_pred": "score_if_playing"})
    df["score_pred"] = df["score_if_playing"] * df["playing_chance"]
    return df


def setup_page(title, icon=None):
    st.set_page_config(page_title=title, page_icon=icon)

    st.markdown(f"# {title}")
    st.sidebar.header(title)


@st.cache
def get_team_data(entry_id, gameweek):
    """Retrieve the gw-by-gw data for a specific entry/team

    credit: vaastav/Fantasy-Premier-League/getters.py

    Args:
        entry_id (int) : ID of the team whose data is to be retrieved
    """
    base_url = "https://fantasy.premierleague.com/api/entry/"
    full_url = base_url + str(entry_id) + "/event/" + str(gameweek) + "/picks/"
    response = requests.get(full_url)
    response.raise_for_status()
    data = response.json()
    return pd.DataFrame(data["picks"])


def get_last_gameweek():
    # TODO
    return 16
