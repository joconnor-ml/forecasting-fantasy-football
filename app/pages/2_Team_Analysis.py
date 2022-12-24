import streamlit as st

import utils

settings = utils.get_settings()


def main():
    utils.setup_page("Team Analysis")
    st.markdown("### Import your team:")
    st.markdown(
        "Copy your team code here from the `points` page on FPL, "
        "e.g. for my team: [https://fantasy.premierleague.com/entry/"
        "**7148592**/event/16](https://fantasy.premierleague.com/entry/7148592/event/16)"
    )
    team_id = int(st.text_input("Team ID", "7148592"))
    team_data = utils.get_team_data(team_id, gameweek=utils.get_current_gameweek())
    team_data = team_data.rename(
        columns={"web_name": "Name", "now_cost": "Price", "event_points": "Points"}
    ).set_index("Name")

    team_data["Price"] /= 10
    st.dataframe(
        team_data.loc[team_data["position"] <= 11, ["Price", "Points"]].style.format(
            {"Price": "{:.1f}"}
        ),
        use_container_width=True,
    )
    st.markdown("Subs")
    st.dataframe(
        team_data.loc[team_data["position"] > 11, ["Price", "Points"]].style.format(
            {"Price": "{:.1f}"}
        ),
        use_container_width=True,
    )

    st.markdown("### Team analysis:")
    st.markdown("Your team is very good, well done.")


if __name__ == "__main__":
    main()
