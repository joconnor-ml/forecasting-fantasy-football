import numpy as np
import pandas as pd
from methodtools import lru_cache


# functions for checking validity of substitution
def sub_possible(position_out, position_in, current_formation_dict):
    if position_out == position_in:
        return True, current_formation_dict
    else:
        formation_dict = current_formation_dict.copy()
        formation_dict[position_out] -= 1
        formation_dict[position_in] += 1
        valid = check_formation_constraints(formation_dict)
        return valid, formation_dict


def check_formation_constraints(formation_dict):
    return (
        formation_dict[1] == 1
        and 3 <= formation_dict[2] <= 5
        and 3 <= formation_dict[3] <= 5
        and 1 <= formation_dict[4] <= 3
        and sum(formation_dict.values()) == 11
    )


class Simulation:
    def __init__(self, selection_df, playing_chance, vice_captain):
        self.selection_df = selection_df
        self.playing_chance = playing_chance
        self.vice_captain = vice_captain

    def run_trials(self, n):
        score_samples = pd.Series(
            [
                self.run_trial(
                    tuple(
                        (
                            self.playing_chance
                            > np.random.uniform(0, 1, self.selection_df.shape[0])
                        ).tolist()
                    )
                )
                for _ in range(n)
            ]
        )
        return score_samples

    @lru_cache(2**16)
    def run_trial(self, is_playing):
        trial_df = self.selection_df.copy()
        trial_df["is_playing"] = is_playing
        trial_df["score_if_playing"] *= trial_df["is_playing"]

        formation = (
            trial_df.loc[trial_df["first_team"], "element_type"]
            .value_counts()
            .to_dict()
        )

        first_team_missing = trial_df[trial_df["first_team"] & ~trial_df["is_playing"]]

        # assume substitution precedence of highest-expected-score first
        available_subs = trial_df[trial_df["sub"] & trial_df["is_playing"]].sort_values(
            "score_if_playing", ascending=False
        )

        # loop through available subs and see if we can legally insert into team
        # I didn't do this particularly efficiently which could be an issue when running
        # large numbers of simulations.
        for i, sub in available_subs.iterrows():
            for j, player in first_team_missing.iterrows():
                valid, new_formation = sub_possible(
                    player["element_type"], sub["element_type"], formation
                )
                if valid:
                    formation = new_formation.copy()
                    trial_df.loc[i, "sub"] = False
                    trial_df.loc[i, "first_team"] = True
                    trial_df.loc[j, "sub"] = True
                    trial_df.loc[j, "first_team"] = False
                    first_team_missing = first_team_missing.drop(j)
                    break

        if not trial_df.loc[trial_df["captain"], "is_playing"].item():
            trial_df.loc[self.vice_captain, "captain"] = True

        total_score = (
            trial_df.loc[trial_df["first_team"], "score_if_playing"].sum()
            + trial_df.loc[trial_df["captain"], "score_if_playing"].sum()
        )
        return total_score


class PlayerScoreSimulation(Simulation):
    def run_trial(self, selection_df, playing_chance=0.95, vice_captain=584):
        trial_df = selection_df.copy().merge(last_year_df[["code", "id"]], on="code", suffixes=("", "_last"))
        trial_df.index = selection_df.index
        trial_df["is_playing"] = playing_chance > np.random.uniform(0, 1, selection_df.shape[0])
        trial_df["rand"] = np.random.uniform(0, 1, trial_df.shape[0])
        trial_df["score"] = \
        pd.merge(trial_df, score_distributions, left_on="id_last", right_on="element").groupby("element").apply(
            lambda grp: grp[grp["p"] > grp["rand"]].iloc[0]
        )["sampled_points"].values

        trial_df["score_if_playing"] = (trial_df["score"] * trial_df["is_playing"]).fillna(0)

        total_points = 0
        formation = trial_df.loc[trial_df["first_team"], "element_type"].value_counts().to_dict()

        first_team_missing = trial_df[trial_df["first_team"] & ~trial_df["is_playing"]]
        missing_by_position = first_team_missing["element_type"].value_counts().to_dict()

        # assume substitution precedence of highest-expected-score first
        available_subs = trial_df[trial_df["sub"] & trial_df["is_playing"]].sort_values("score_if_playing",
                                                                                        ascending=False)

        # loop through available subs and see if we can legally insert into team
        # I didn't do this particularly efficiently which could be an issue when running
        # large numbers of simulations.
        for i, sub in available_subs.iterrows():
            for j, player in first_team_missing.iterrows():
                valid, new_formation = sub_possible(player["element_type"], sub["element_type"], formation)
                if valid:
                    formation = new_formation.copy()
                    trial_df.loc[i, "sub"] = False
                    trial_df.loc[i, "first_team"] = True
                    trial_df.loc[j, "sub"] = True
                    trial_df.loc[j, "first_team"] = False
                    first_team_missing = first_team_missing.drop(j)
                    break

        if not trial_df.loc[trial_df["captain"], "is_playing"].item():
            trial_df.loc[vice_captain, "captain"] = True

        total_score = (
                trial_df.loc[trial_df["first_team"], "score_if_playing"].sum() +
                trial_df.loc[trial_df["captain"], "score_if_playing"].sum()
        )
        return total_score
