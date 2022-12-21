import streamlit as st

import utils

settings = utils.get_settings()


def main():
    utils.setup_page("Team Optimiser")
    team_id = int(st.text_input("Team ID", "7148592"))
    team_data = utils.get_team_data(team_id, gameweek=utils.get_last_gameweek())
    st.dataframe(team_data)


if __name__ == "__main__":
    main()
