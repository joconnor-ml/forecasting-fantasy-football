def get_new_players(df, last_year_df):
    return df[~df["player_indices"].isin(last_year_df["player_indices"])].index()

def get_transferred_players(df, last_year_df):
    pass