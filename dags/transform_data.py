"""Get data into nice form for training
our models"""

import os

import numpy as np
import pandas as pd
import pymongo


def transform_data(execution_date, **kwargs):
    client = pymongo.MongoClient(os.environ['MONGO_URL'])
    db = client["fantasy_football"]
    
    player_details = pd.DataFrame(list(db["elements"].find()))
    player_details.index = player_details.id
    player_details = player_details[["team_code", "web_name", "element_type"]]

    player_history = db["player_data"].find({}, {"history": 1, "history_past": 1})
    player_dfs = {}
    for i, player in enumerate(player_history):
        df = pd.DataFrame(player["history"])
        df = df.reset_index()
        df.index += df["round"].min()
        player_df = df[["minutes", "bps", "total_points", "was_home", "opponent_team", "season"]].astype(np.float64)
        player_df.loc[:, "appearances"] = (player_df.loc[:, "minutes"] > 0).astype(np.float64)
        mean3 = player_df[["total_points", "minutes", "bps", "appearances"]].rolling(3).mean()
        mean10 = player_df[["total_points", "minutes", "bps", "appearances"]].rolling(10).mean()
        std10 = player_df[["total_points"]].rolling(5).std()
        mean5 = player_df[["total_points", "minutes", "bps", "appearances"]].rolling(5).mean()
        ewma = player_df[["total_points", "minutes", "bps", "appearances"]].ewm(halflife=10).mean()
        cumulative_sums = player_df.cumsum(axis=0)
        # normalise by number of games played up to now
        cumulative_means = cumulative_sums[["total_points", "minutes", "bps", "appearances"]].div(cumulative_sums.loc[:, "appearances"] + 1, axis=0)
        player_df["id"] = df["element"]
        # join on player details to get position ID, name and team ID.
        player_df = pd.merge(player_df, player_details,
                             how="left", left_on="id", right_index=True)
        player_df["target"] = player_df["total_points"].shift(-1)
        player_df["target_minutes"] = player_df["minutes"].shift(-1)
        player_df["target_home"] = player_df["was_home"].shift(-1)
        player_df["target_team"] = player_df["opponent_team"].shift(-1)
        player_df["gameweek"] = player_df.index

        #one_hot = True
        
        #if one_hot:
        # apply one-hot encoding to categorical variables
        #opponent_team = pd.get_dummies(player_df["target_team"]).add_prefix("opponent_")
        #own_team = pd.get_dummies(player_df["team_code"]).add_prefix("team_")
        #position = pd.get_dummies(player_df["element_type"]).add_prefix("position_")
        #player_df = pd.concat([player_df.drop(["target_team", "team_code", "element_type"], axis=1),
        #                           opponent_team, own_team, position], axis=1)

        past_seasons = player["history_past"]
        if past_seasons:
            player_df["last_season_points"] = past_seasons[-1]["total_points"]
            player_df["last_season_minutes"] = past_seasons[-1]["minutes"]
            player_df["last_season_ppm"] = player_df["last_season_points"] / player_df["last_season_minutes"]

        player_dfs[i] = pd.concat([
            player_df,
            (mean3 - cumulative_means).add_suffix("_mean3"),
            (mean5 - cumulative_means).add_suffix("_mean5"),
            (mean10 - cumulative_means).add_suffix("_mean10"),
            std10.add_suffix("_std10"),
            (ewma - cumulative_means).add_suffix("_ewma"),
            cumulative_means.add_suffix("_mean_all"),
            cumulative_sums.add_suffix("_sum_all"),
        ], axis=1)

    player_df = pd.concat(player_dfs)
    player_df.to_csv("/data/test_data.csv")

    team_data = player_df.groupby(["team_code",
                                   "gameweek"]).sum().reset_index()
    team_pos_data = player_df.groupby(["team_code", "element_type",
                                       "gameweek"]).sum().reset_index()

    cumsums = team_data.groupby(["team_code"]).shift().cumsum()
    cummeans = cumsums.div(cumsums["appearances"], axis=0)
    cummeans["team_code"] = team_data["team_code"]
    cummeans["gameweek"] = team_data["gameweek"]
    player_df = pd.merge(player_df, cummeans[["team_code", "gameweek",
                                              "total_points"]],
                         how="left",
                         on=["team_code", "gameweek"],
                         suffixes=("", "_team_mean"))

    cumsums = sum(team_data.groupby(["team_code"]).shift(i) for i in range(1,4))
    cummeans = cumsums.div(cumsums["appearances"], axis=0)
    cummeans["team_code"] = team_data["team_code"]
    cummeans["gameweek"] = team_data["gameweek"]
    player_df = pd.merge(player_df, cummeans[["team_code", "gameweek",
                                              "total_points"]],
                         how="left",
                         on=["team_code", "gameweek"],
                         suffixes=("", "_team_last3"))

    team_pos_data.to_csv("/data/team_pos_data.csv")
    cumsums = team_pos_data.groupby(["team_code",
                                     "element_type"]).shift().cumsum()
    cummeans = cumsums.div(cumsums["appearances"], axis=0)
    cummeans["team_code"] = team_pos_data["team_code"]
    cummeans["element_type"] = team_pos_data["element_type"]
    cummeans["gameweek"] = team_pos_data["gameweek"]
    player_df = pd.merge(player_df, cummeans[["team_code", "element_type",
                                              "gameweek", "total_points"]],
                         how="left",
                         on=["team_code", "element_type", "gameweek"],
                         suffixes=("", "_team_pos_mean"))
    cumsums = sum(team_pos_data.groupby(["team_code",
                                         "element_type"]).shift(i) for i in range(1, 4))
    cummeans = cumsums.div(cumsums["appearances"], axis=0)
    cummeans["team_code"] = team_pos_data["team_code"]
    cummeans["element_type"] = team_pos_data["element_type"]
    cummeans["gameweek"] = team_pos_data["gameweek"]
    player_df = pd.merge(player_df, cummeans[["team_code", "element_type",
                                              "gameweek", "total_points"]],
                         how="left",
                         on=["team_code", "element_type", "gameweek"],
                         suffixes=("", "_team_pos_last3"))

    weight = player_df["appearances_sum_all"] / (player_df["appearances_sum_all"] + 30)
    player_df["bayes_global"] = (player_df["total_points_mean_all"].fillna(3.5) * weight) + (3.5 * (1 - weight))
    # subtracting bayes_global from the next two to combat multicollinearity
    player_df["bayes_team"] = (player_df["total_points_team_mean"].fillna(3.5) * weight) + (3.5 * (1 - weight)) - player_df["bayes_global"]
    player_df["bayes_team_pos"] = (player_df["total_points_team_pos_mean"].fillna(3.5) * weight) + (3.5 * (1 - weight)) - player_df["bayes_global"]

    # need to look up the current fixture data for future predictions
    teams = []
    for team in db["teams"].find():
        team["next_opponent"] = team["next_event_fixture"][0]["opponent"]
        team["is_home"] = team["next_event_fixture"][0]["is_home"]
        teams.append(team)
    teams = pd.DataFrame(teams)
    teams.index = teams.code

    last_gameweek = player_df["gameweek"].max()
    last_season = player_df["season"].max()
    last_week_df = player_df.loc[(player_df["gameweek"] == last_gameweek) &
                                 (player_df["season"] == last_season)]
    last_week_teams = teams.loc[last_week_df["team_code"]]
    player_df.loc[(player_df["gameweek"] == last_gameweek) &
                  (player_df["season"] == last_season), "target_team"] = last_week_teams["next_opponent"].values
    player_df.loc[(player_df["gameweek"] == last_gameweek) &
                  (player_df["season"] == last_season), "target_home"] = last_week_teams["is_home"].values.astype(int)

    player_df.to_csv("/data/data.csv")



if __name__ == "__main__":
    import datetime

    transform_data(datetime.datetime.now())
