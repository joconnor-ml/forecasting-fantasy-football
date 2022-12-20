import numpy as np

from fpl_forecast import total_points, playing_chance
from fpl_forecast import utils as forecast_utils

TASKS = {"total_points": total_points, "playing_chance": playing_chance}
SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23"]


def main():
    for name, module in TASKS.items():
        print(f"Running task {name}")
        df = forecast_utils.get_player_data(seasons=SEASONS)
        # predict total points over next 5 weeks
        targets = module.get_targets(df)
        features = module.generate_features(df)

        for i, grp in df.groupby("position"):
            (
                train_features,
                val_features,
                train_targets,
                val_targets,
            ) = module.train_test_split(grp, features, targets)

            ## benchmark:
            benchmark_pred = np.ones_like(val_targets) * val_targets.mean()
            module.get_scores(benchmark_pred, val_targets)

            for model in module.get_models():
                model, preds = module.test_model(
                    model, train_features, train_targets, val_features, val_targets
                )
                print(model)

            forecast_df = grp[
                (df["GW"] == df["GW"].max()) & (df["season"] == SEASONS[-1])
            ]
            features = module.generate_features(forecast_df)

            forecast_df["p"] = module.predict(model, features)

            out_df = forecast_df.sort_values("value")[
                [
                    "code",
                    "name",
                    "team",
                    "position",
                    "value",
                    "value_rank",
                    "minutes",
                    "total_points",
                    "p",
                ]
            ]
            out_df["price"] = out_df["value"] / 10
            out_df.to_parquet(f"{name}.pq", index=False)


if __name__ == "__main__":
    main()
