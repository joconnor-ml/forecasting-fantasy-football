"""Get data into nice form for training
our models"""

from os.path import dirname, join

import numpy as np
import pandas as pd
import pymongo

def generate_features(execution_date, **kwargs):
    player_df = df[["minutes", "total_points", "was_home", "opponent_team"]].astype(np.float64)
    player_df.loc[:, "appearances"] = (player_df.loc[:, "minutes"] > 0).astype(np.float32)
    mean3 = player_df[["total_points", "minutes"]].rolling(3).mean()
    mean10 = player_df[["total_points", "minutes"]].rolling(10).mean()
    cumulative_sums = player_df[["appearances", "total_points"]].cumsum(axis=0)
    # normalise by number of games played up to now
    cumulative_means = cumulative_sums[["total_points"]].div(cumulative_sums.loc[:, "appearances"] + 1, axis=0)
    
    df = pd.concat([player_df,
                    mean3.add_suffix("_mean3"),
                    mean10.add_suffix("_mean10"),
                    cumulative_means.add_suffix("_mean_all")], axis=1)
