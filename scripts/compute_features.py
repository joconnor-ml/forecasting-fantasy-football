import pandas as pd

from fpl_forecast import total_points
from fpl_forecast import utils as forecast_utils


def main(position: str, horizon: int):
    df = forecast_utils.get_player_data(seasons=forecast_utils.TRAIN_SEASONS)
    df = df[
        ((df["position"] == position) & (df["minutes"] > 0))
        | (
            (df["season"] == forecast_utils.TRAIN_SEASONS[-1])
            & (df["position"] == position)
            & (
                df["GW"]
                == df[
                    (df["season"] == forecast_utils.TRAIN_SEASONS[-1])
                    & df["total_points"].notnull()
                ]["GW"].max()
            )
        )
    ]

    existing_features = []
    for model_name, model in total_points.get_models(position, horizon).items():
        features = model.generate_features(df)
        targets = model.get_targets(df)
        pd.concat([features, targets], axis=1).to_parquet(
            f"prod/features/{position}_{horizon}.pq"
        )
        break


if __name__ == "__main__":
    max_horizon = 5
    for position in ["GK", "DEF", "MID", "FWD"]:
        for horizon in range(1, max_horizon + 1):
            main(position, horizon)
