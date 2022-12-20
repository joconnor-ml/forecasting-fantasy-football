from functools import lru_cache

import pandas as pd
import streamlit as st
from pydantic import BaseSettings


class Settings(BaseSettings):
    bucket_name: str = "forecasting-fantasy-football"
    points_data_path: str = "gs://forecasting-fantasy-football/prod/points.pq"
    playing_data_path: str = "gs://forecasting-fantasy-football/prod/playing.pq"
    forecast_horizon: int = 5

    class Config:
        env_file = ".env"


@lru_cache
def get_settings():
    return Settings()


@st.cache
def read_parquet_cached(path):
    df = pd.read_parquet(path)
    return df


@st.cache
def get_forecast_data(points_path, playing_path):
    playing = read_parquet_cached(playing_path)
    points = read_parquet_cached(points_path)[["name", "team", "score_pred", "horizon"]]
    df = playing.merge(points, how="left", on=["name", "team", "horizon"]).set_index(
        "name"
    )[["horizon", "score_pred", "playing_chance"]]
    df = df.rename(columns={"score_pred": "score_if_playing"})
    df["score_pred"] = df["score_if_playing"] * df["playing_chance"]
    return df
