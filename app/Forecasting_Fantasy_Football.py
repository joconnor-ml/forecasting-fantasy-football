import streamlit as st

import utils

utils.setup_page(
    "Forecasting Fantasy Football",
    icon="⚽",
)

current_gameweek = utils.get_current_gameweek()
st.markdown(
    f"""
    Current gameweek: {current_gameweek}
    
    Source code on [Github](https://github.com/joconnor-ml/forecasting-fantasy-football.git)
"""
)
