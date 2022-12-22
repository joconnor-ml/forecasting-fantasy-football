import streamlit as st

import utils

st.set_page_config(
    page_title="Forecasting Fantasy Football",
    page_icon="⚽",
)

st.write("# Forecasting Fantasy Football")

current_gameweek = utils.get_current_gameweek()
st.markdown(
    f"""
    Current gameweek: {current_gameweek}
    
    Source code on [Github](https://github.com/joconnor-ml/forecasting-fantasy-football.git)
"""
)
