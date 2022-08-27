import pandas as pd

PLAYER_ID_COL = "code"


def generate_targets(df, horizon, target_cols=("total_points", "minutes")):
    return df.groupby(PLAYER_ID_COL)[target_cols].shift(-horizon)


def generate_lag_features(df, cols, lags=(0, 1, 2)):
    feats = (
        df.groupby(PLAYER_ID_COL)[cols].shift(lag).add_suffix(f"_lag_{lag}")
        for lag in lags
    )
    return pd.concat(feats, axis=1)


def generate_rolling_features(df, cols, windows=(3, 10, 19), aggs=("mean", "median")):
    feats = (
        df.groupby(PLAYER_ID_COL)[cols]
        .rolling(window)
        .agg(agg)
        .add_suffix(f"_rolling_{window}_{agg}")
        for window in windows
        for agg in aggs
    )
    return pd.concat(feats, axis=1).reset_index(PLAYER_ID_COL, drop=True).sort_index()


def get_player_data(seasons):
    df = (
        pd.concat(
            [
                pd.read_csv(
                    f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/gws/merged_gw.csv"
                )
                .merge(
                    pd.read_csv(
                        f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/players_raw.csv",
                        usecols=["id", "code"],
                    ),
                    left_on="element",
                    right_on="id",
                    how="left",
                )
                .assign(season=season)
                for season in seasons
            ]
        )
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
