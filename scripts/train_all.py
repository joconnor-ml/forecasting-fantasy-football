from . import train_points_model
import pandas as pd
import pathlib
def main(max_horizon, output_path):
    all_preds = []
    for position in ["GK", "DEF", "MID", "FWD"]:
        for horizon in range(1, max_horizon+1):
            pred_df = train_points_model.main(position, horizon)
            pred_df["horizon"] = horizon
            all_preds.append(pred_df)
    pd.concat(all_preds).to_csv(output_path)
