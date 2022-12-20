import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="⚽",
)

st.write("# Forecasting Fantasy Football")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    [Github](https://github.com/joconnor-ml/forecasting-fantasy-football.git)
"""
)
