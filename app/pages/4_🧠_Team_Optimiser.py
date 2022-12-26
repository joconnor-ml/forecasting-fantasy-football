import streamlit as st

import utils

settings = utils.get_settings()


def main():
    budget = float(st.text_input("Budget", "100"))
    selection_df = utils.pick_team(settings.points_data_path, settings.playing_data_path, settings.bucket_name, budget)
    st.dataframe(
        selection_df,
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
