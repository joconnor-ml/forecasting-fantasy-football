import pandas as pd

PLAYER_ID_COL = "code"


def generate_targets(df, target_col, horizon):
    return df.groupby(PLAYER_ID_COL)[target_col].shift(-horizon)


def generate_lag_features(df, cols, lags=(0, 1, 2)):
    feats = (
        df.groupby(PLAYER_ID_COL)[cols].shift(lag).add_suffix(f"lag_{lag}")
        for lag in lags
    )
    pd.concat(feats, axis=1)


def generate_rolling_features(df, cols, windows=(3, 10, 19), aggs=("mean", "median")):
    feats = (
        df.groupby(PLAYER_ID_COL)[cols]
        .rolling(window)
        .agg(agg)
        .add_suffix(f"rolling_{window}_{agg}")
        for window in windows
        for agg in aggs
    )
    return pd.concat(feats, axis=1)


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
