"""NB: There is some (probably mild) overfitting in here since we are using
team and position means over the whole training set.
"""
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np

# going to do some hand-modelling.

# treat average player score as a fixed hidden variable to be estimated.

# at the start of the season we have no information about how good the player is. Start with a general prior based on the distribution of scores for players in this position.

# each game contributes evidence.

# wait up

# assume no prior

# after one game, player's expected score is their last score. Std dev of this
# is undefined?

# after two games, players expected score is the avg. Std dev is std dev of
# distribution / sqrt(2)

# in general, expected score is normally distributed with mu = mean score,
# sigma = std dev of scores / sqrt(n)

class MeanPointsRegressor(BaseEstimator):
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        return X["total_points_mean_all"]

class BayesianPointsRegressor(BaseEstimator):
    def __init__(self, prior="global"):
        """Estimator that predicts score as a weighted average of
        some prior expectation and the player's points per game. The weight
        given to points per game is a function of the number of games played
        and the weight_par. Higher weight pars give more weight to the prior.
        
        arguments
        =========
        prior: str, can be one of "global", "team", "position". The player's
        prior points expectation is set to the global mean score, team mean or
        position mean.
        """
        self.prior = prior

    def _predict(self, X, weight_par):
        n = X["appearances_sum_all"]
        weight = n / (n + weight_par)
        if self.prior == "global":
            prior_score = self.overall_mean
        if self.prior == "team":
            prior_score = pd.merge(X, self.team_means.to_frame(),
                                   left_on="team_code", right_index=True, how="left")["target"].fillna(self.overall_mean)
            
        elif self.prior == "position":
            prior_score = pd.merge(X, self.position_means.to_frame(),
                                   left_on="element_type", right_index=True, how="left")["target"].fillna(self.overall_mean)
        return X["total_points_mean_all"] * weight + prior_score * (1 - weight)

    def fit(self, X, y):
        self.overall_mean = y.mean()
        if self.prior == "team":
            self.team_means = y.groupby(X["team_code"]).mean()
        if self.prior == "position":
            self.position_means = y.groupby(X["element_type"]).mean()
        # grid search over weight par
        weights = np.logspace(1, 2, 20)
        scores = []
        for weight in weights:
            preds = self._predict(X, weight)
            scores.append(((preds - y)**2).sum())
        scores = pd.Series(scores, index=weights)
        self.weight_par = scores.idxmin()
        return self

    def predict(self, X):
        return self._predict(X, self.weight_par)

