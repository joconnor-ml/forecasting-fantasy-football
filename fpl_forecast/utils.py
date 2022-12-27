import pandas as pd
from loguru import logger

PLAYER_ID_COL = "code"
SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23"]
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
        .ewm(halflife=window)
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

    fixture_df = get_fixture_df(seasons)
    df = df.merge(
        fixture_df[["season", "id", "team", "total_difficulty", "opponent"]],
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
                ["season", "id", "team", "total_difficulty", "event", "was_home"]
            ],
            left_on=["team", "GW"],
            right_on=["team", "event"],
        )
        for i in range(int(next_gw), int(fixture_df.event.max()))
    )
    df["xP"] = df["xP"].fillna(df["total_points"])
    df = pd.concat([df, future_players.assign(minutes=90)]).reset_index()

    return df


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
    fixtures = pd.concat(
        [
            fixtures.rename(
                columns={
                    "team_h": "team",
                    "team_h_difficulty": "difficulty",
                    "team_h_score": "score",
                    "team_a": "opponent",
                    "team_a_difficulty": "opponent_difficulty",
                    "team_a_score": "opponent_score",
                }
            ).assign(was_home=True),
            fixtures.rename(
                columns={
                    "team_a": "team",
                    "team_a_difficulty": "difficulty",
                    "team_a_score": "score",
                    "team_h": "opponent",
                    "team_h_difficulty": "opponent_difficulty",
                    "team_h_score": "opponent_score",
                }
            ).assign(was_home=False),
        ]
    ).reset_index()
    fixtures["margin"] = fixtures["score"] - fixtures["opponent_score"]
    fixtures["total_difficulty"] = (
        fixtures["opponent_difficulty"] - fixtures["difficulty"]
    )
    return fixtures
