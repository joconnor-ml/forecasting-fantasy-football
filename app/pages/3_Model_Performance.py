import streamlit as st

import utils

settings = utils.get_settings()


def main():
    utils.setup_page("Model Performance")
    points_scores = utils.read_parquet_cached(settings.points_models_data, settings.bucket_name)
    st.dataframe(data=points_scores, use_container_width=True)

    # playing_scores = utils.read_parquet_cached(settings.playing_models_data)


if __name__ == "__main__":
    main()
