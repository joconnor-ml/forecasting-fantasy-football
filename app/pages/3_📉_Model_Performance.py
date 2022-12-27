import pandas as pd
import plotly.express as px
import streamlit as st

import utils

settings = utils.get_settings()


@st.cache
def get_model_scores(points_models_data, playing_models_data, bucket_name):
    points_scores = utils.read_parquet_cached(points_models_data, bucket_name)
    playing_scores = utils.read_parquet_cached(playing_models_data, bucket_name)
    scores_by_model_type = pd.concat(
        [
            points_scores.groupby(["position", "horizon"])[["rmse"]]
            .min()
            .reset_index()
            .rename(columns={"rmse": "score", "position": "model"}),
            playing_scores.groupby(["horizon"])[["accuracy"]]
            .max()
            .reset_index()
            .rename(columns={"accuracy": "score"})
            .assign(model="playing_chance"),
        ]
    )
    return scores_by_model_type, points_scores, playing_scores


def main():
    utils.setup_page("Model Performance", icon="📉")
    best_scores, points_scores, playing_scores = get_model_scores(
        settings.points_models_data, settings.playing_models_data, settings.bucket_name
    )

    st.markdown("### Playing Chance Accuracy")
    fig = px.line(
        best_scores[best_scores["model"] == "playing_chance"],
        x="horizon",
        y="score",
        hover_data=["score"],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Gameweek Points RMSE by Position")
    fig = px.line(
        best_scores[best_scores["model"] != "playing_chance"],
        x="horizon",
        y="score",
        color="model",
        hover_data=["score", "model"],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### All Models: Playing Chance")
    st.dataframe(playing_scores, use_container_width=True)

    st.markdown("### All Models: Gameweek Points")
    st.dataframe(points_scores, use_container_width=True)

    st.markdown("### Feature importances: Gameweek Points")
    feature_importance = (
        utils.read_parquet_cached(settings.feature_imps_path, settings.bucket_name)
        .reset_index()
        .groupby(["index", "position"])
        .mean()
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    fig = px.bar(
        feature_importance,
        x="feature",
        y="importance",
        color="position",
        barmode="group",
    )
    st.plotly_chart(fig, use_container_width=True)

    feature_importance = (
        utils.read_parquet_cached(settings.feature_imps_path, settings.bucket_name)
        .reset_index()
        .groupby(["index", "horizon"])
        .mean()
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    feature_importance["horizon"] = feature_importance["horizon"].astype(str)
    fig = px.bar(
        feature_importance,
        x="feature",
        y="importance",
        color="horizon",
        barmode="group",
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
