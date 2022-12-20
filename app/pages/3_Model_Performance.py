import streamlit as st

import utils

settings = utils.get_settings()


def main():
    st.set_page_config(page_title="Team Optimiser", page_icon="📈")

    st.markdown("# Team Optimiser")
    st.sidebar.header("Team Optimiser")

    points_scores = utils.read_parquet_cached(settings.points_models_data)
    st.dataframe(data=points_scores, use_container_width=True)

    #playing_scores = utils.read_parquet_cached(settings.playing_models_data)


if __name__ == "__main__":
    main()
