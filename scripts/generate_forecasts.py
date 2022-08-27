import pandas as pd
from fpl_forecast import utils as forecast_utils
from fpl_forecast import total_points, playing_chance

TASKS = {"total_points": total_points, "playing_chance": playing_chance}


def main():
    for name, module in TASKS.items():
        print(f"Running task {name}")
        df = forecast_utils.get_player_data(seasons=["2020-21", "2021-22"])
        # predict total points over next 5 weeks
        targets = module.get_targets(df)
        features = module.generate_features(df)
        (
            train_features,
            val_features,
            train_targets,
            val_targets,
        ) = module.train_test_split(df, features, targets)

        ## benchmark:
        benchmark_pred = pd.np.ones_like(val_targets) * val_targets.mean()
        module.get_scores(benchmark_pred, val_targets)

        for model in module.get_models():
            model, preds = module.test_model(
                model, train_features, train_targets, val_features, val_targets
            )
            print(model)

        forecast_df = forecast_utils.get_player_data(["2022-23"])
        latest_week = forecast_df["GW"].max()
        features = module.generate_features(forecast_df)

        forecast_df["p"] = module.predict(model, features)

        forecast_df.query(f"GW=={latest_week}").sort_values("value")[
            [
                "name",
                "team",
                "position",
                "value",
                "value_rank",
                "minutes",
                "total_points",
                "p",
            ]
        ].to_csv(f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
