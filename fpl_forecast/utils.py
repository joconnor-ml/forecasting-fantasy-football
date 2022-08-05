def get_new_players(df, last_year_df):
    return df[~df["code"].isin(last_year_df["code"])].index()

def get_transferred_players(df, last_year_df):
    pass