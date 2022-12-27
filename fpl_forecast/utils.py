from collections import defaultdict

import pandas as pd
from loguru import logger

PLAYER_ID_COL = "code"
TRAIN_SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23"]
ALL_SEASONS = ["2018-19"] + TRAIN_SEASONS
GW_COLS = [
    "name",
    "position",
    "team",
    "xP",
    "assists",
    "bonus",
    "bps",
    "clean_sheets",
    "creativity",
    "element",
    "fixture",
    "goals_conceded",
    "goals_scored",
    "ict_index",
    "influence",
    "kickoff_time",
    "minutes",
    "opponent_team",
    "own_goals",
    "penalties_missed",
    "penalties_saved",
    "red_cards",
    "round",
    "saves",
    "selected",
    "team_a_score",
    "team_h_score",
    "threat",
    "total_points",
    "transfers_balance",
    "transfers_in",
    "transfers_out",
    "value",
    "was_home",
    "yellow_cards",
    "GW",
]


def generate_targets(df, horizon, target_cols=("total_points", "minutes")):
    return df.groupby(PLAYER_ID_COL)[target_cols].shift(-horizon)


def generate_lag_features(df, cols, lags=(0, 1, 2)):
    feats = (
        df.groupby(PLAYER_ID_COL)[cols].shift(lag).add_suffix(f"_lag_{lag}")
        for lag in lags
    )
    return pd.concat(feats, axis=1)


def generate_rolling_features(df, cols, windows=(3, 19), aggs=("mean", "median")):
    feats = (
        df.groupby(PLAYER_ID_COL)[cols]
        .ewm(halflife=window / 2, min_periods=window // 2)
        .agg(agg)
        .add_suffix(f"_rolling_{window}_{agg}")
        for window in windows
        for agg in aggs
    )
    return pd.concat(feats, axis=1).reset_index(PLAYER_ID_COL, drop=True).sort_index()


def get_gw_data(season):
    dfs = []
    for i in range(1, 39):
        try:
            df = pd.read_csv(
                f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/gws/gw{i}.csv",
                usecols=lambda x: x in GW_COLS,
            )
            dfs.append(df)
        except Exception as e:
            logger.info(e)
            break
    return pd.concat(dfs).rename(columns={"team": "team_name"})


def get_player_data(seasons):
    df = (
        pd.concat(
            [
                get_gw_data(season)
                .merge(
                    pd.read_csv(
                        f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/players_raw.csv",
                        usecols=[
                            "id",
                            "code",
                            "team",
                            "team_code",
                            "selected_by_percent",
                        ],
                    ),
                    left_on="element",
                    right_on="id",
                    how="left",
                )
                .assign(season=season)
                for season in seasons
            ]
        )
        .rename(columns={"round": "GW"})
        .sort_values(["season", "GW"])
        .reset_index(drop=True)
    )

    df["played"] = (df["minutes"] > 0).astype(int)
    df["position"] = df["position"].replace({"GKP": "GK"})
    df = df.join(pd.get_dummies(df["position"]))

    df["value_rank"] = df.groupby(["team", "position", "GW", "season"])["value"].rank(
        "dense", ascending=False
    )
    df["team_size"] = df.groupby(["team", "position", "GW", "season"])[
        "value"
    ].transform("size")

    fixture_df = get_fixture_df(ALL_SEASONS)  # add extra seasons to improve elo
    df = df.merge(
        fixture_df[
            [
                "season",
                "id",
                "team",
                "total_difficulty",
                "opponent",
                "elo",
                "opponent_elo",
                "win_prob",
                "elo_diff"
            ]
        ],
        left_on=["season", "fixture", "team"],
        right_on=["season", "id", "team"],
        how="left",
    )

    # extend current season into future using fixture df
    this_season = df.query(f"season=='{seasons[-1]}'")
    next_gw = this_season["GW"].max() + 1
    players = (
        this_season.groupby("code")[["name", "position", "team"]].last().reset_index()
    )
    future_players = pd.concat(
        players.assign(GW=i).merge(
            fixture_df.query(f"season=='{seasons[-1]}'")[
                ["season", "id", "team", "total_difficulty", "event", "was_home", "elo_diff", "win_prob"]
            ],
            left_on=["team", "GW"],
            right_on=["team", "event"],
        )
        for i in range(int(next_gw), int(fixture_df.event.max()))
    )
    df["xP"] = df["xP"].fillna(df["total_points"])
    df = pd.concat([df, future_players.assign(minutes=90)]).reset_index()

    return df


def calculate_elo(fixtures):
    HOME_ADVANTAGE = 50
    def expect_result(p1, p2):
        exp = (p2 - p1) / 400.0
        home_win = 1 / ((10.0 ** (exp)) + 1)
        return home_win, 1 - home_win

    def get_score_weight(home_score, away_score):
        score_diff = abs(home_score - away_score)
        if score_diff < 2:
            score_weight = 1
        elif score_diff == 2:
            score_weight = 3 / 2
        else:
            score_weight = (11 + score_diff) / 8
        return score_weight

    def get_result(home_score, away_score):
        if home_score > away_score:
            home_weight = 1
            away_weight = 0
        elif home_score < away_score:
            home_weight = 0
            away_weight = 1
        else:
            home_weight = away_weight = 0.5
        return home_weight, away_weight

    def update(ratings, home, away, home_score, away_score, home_advantage, k):
        pred_home_win, pred_away_win = expect_result(
            ratings[home] + home_advantage, ratings[away]
        )
        home_result, away_result = get_result(home_score, away_score)
        score_weight = get_score_weight(home_score, away_score)
        home_update = k * score_weight * (home_result - pred_home_win)
        away_update = k * score_weight * (away_result - pred_away_win)
        return home_update, away_update

    ratings = defaultdict(lambda: 1300.0)  # dict of {team: rating}
    output = []
    for i, row in fixtures.iterrows():
        output.append(
            dict(
                team_h_elo=ratings[row["team_h"]],
                team_a_elo=ratings[row["team_a"]],
                home_win_prob=expect_result(ratings[row["team_h"]] + HOME_ADVANTAGE, ratings[row["team_a"]])[0],
            )
        )
        if row[["team_h_score", "team_a_score"]].isnull().any(): continue
        home_update, away_update = update(
            ratings,
            row["team_h"],
            row["team_a"],
            row["team_h_score"],
            row["team_a_score"],
            home_advantage=HOME_ADVANTAGE,
            k=40,
        )
        ratings[row["team_h"]] += home_update
        ratings[row["team_a"]] += away_update
    return pd.DataFrame(output, index=fixtures.index)


def get_score_distributions():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2021-22/gws/merged_gw.csv"
    )
    score_distributions = (
        df[df["minutes"] > 0]
        .groupby("element")["total_points"]
        .value_counts(normalize=True)
    )
    score_distributions = (
        score_distributions.groupby(level=0)
        .cumsum()
        .to_frame("p")
        .reset_index()
        .rename({"total_points": "sampled_points"}, axis=1)
    )
    return score_distributions


def get_fixture_df(seasons):
    fixtures = pd.concat(
        pd.read_csv(
            f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/fixtures.csv"
        ).assign(season=season)
        for season in seasons
    )
    fixtures = pd.concat([fixtures, calculate_elo(fixtures)], axis=1)
    fixtures = pd.concat(
        [
            fixtures.rename(
                columns={
                    "team_h": "team",
                    "team_h_difficulty": "difficulty",
                    "team_h_score": "score",
                    "team_h_elo": "elo",
                    "home_win_prob": "win_prob",
                    "team_a": "opponent",
                    "team_a_difficulty": "opponent_difficulty",
                    "team_a_score": "opponent_score",
                    "team_a_elo": "opponent_elo",
                }
            ).assign(was_home=True),
            fixtures.rename(
                columns={
                    "team_a": "team",
                    "team_a_difficulty": "difficulty",
                    "team_a_score": "score",
                    "team_a_elo": "elo",
                    "team_h": "opponent",
                    "team_h_difficulty": "opponent_difficulty",
                    "team_h_score": "opponent_score",
                    "team_h_elo": "opponent_elo",
                }
            ).assign(was_home=False, win_prob=lambda x: 1 - x.home_win_prob),
        ]
    ).reset_index()
    fixtures["margin"] = fixtures["score"] - fixtures["opponent_score"]
    fixtures["total_difficulty"] = (
        fixtures["opponent_difficulty"] - fixtures["difficulty"]
    )
    fixtures["elo_diff"] = fixtures["elo"] - fixtures["opponent_elo"]
    return fixtures
