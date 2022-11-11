import streamlit as st
import pandas as pd

POINTS_DATA_PATH = r"C:\Users\Joe\Downloads\points (1)\points.csv"
PLAYING_DATA_PATH = r"C:\Users\Joe\Downloads\playing (1)\playing.csv"
MAX_HORIZONS = 4

@st.cache
def get_points_data():
    df = pd.read_csv(POINTS_DATA_PATH)
    return df

@st.cache
def get_playing_data():
    df = pd.read_csv(PLAYING_DATA_PATH)
    return df


def main():
    playing = get_playing_data()
    points = get_points_data()[["name", "team", "score_pred", "horizon"]]
    df = playing.merge(points, how="left", on=["name", "team", "horizon"]).set_index("name")[["horizon", "score_pred", "playing_chance"]]
    df = df.rename(columns={"score_pred": "score_if_playing"})
    df["score_pred"] = df["score_if_playing"] * df["playing_chance"]

    players = st.multiselect(
        "Choose players", list(df.index.unique())
    )
    if not players:
        st.error("Please select at least one player.")
    horizons = st.multiselect(
        "Choose horizons", range(1, MAX_HORIZONS+1)
    )
    if not horizons:
        st.error("Please select at least one horizon.")
    else:
        st.write("### Players", df[df.horizon.isin(horizons)].loc[players])


if __name__ == "__main__":
    main()