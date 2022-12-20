import streamlit as st

import utils

settings = utils.get_settings()

def main():
    st.set_page_config(page_title="Team Optimiser", page_icon="📈")

    st.markdown("# Team Optimiser")
    st.sidebar.header("Team Optimiser")


if __name__ == "__main__":
    main()
