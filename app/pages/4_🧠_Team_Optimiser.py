import streamlit as st

import fpl_opt
import utils

settings = utils.get_settings()


def main():
    budget = float(st.text_input("Budget", "100"))
    selection_df = fpl_opt.select_new_team.main(budget)
    st.dataframe(
        selection_df,
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
