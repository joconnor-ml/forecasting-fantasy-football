import pulp
import numpy as np


position_data = {
    "gk": {"position_id": 1, "min_players": 1, "max_players": 1, "num_total": 2},
    "df": {"position_id": 2, "min_players": 3, "max_players": 5, "num_total": 5},
    "mf": {"position_id": 3, "min_players": 3, "max_players": 5, "num_total": 5},
    "fw": {"position_id": 4, "min_players": 1, "max_players": 3, "num_total": 3},
}


def get_decision_array(name, length):
    return [
        pulp.LpVariable("{}_{}".format(name, i), lowBound=0, upBound=1, cat='Integer')
        for i in range(length)
    ]


class TransferOptimiser:
    def __init__(self, expected_scores, buy_prices, sell_prices, positions, clubs):
        self.expected_scores = expected_scores
        self.buy_prices = buy_prices
        self.sell_prices = sell_prices
        self.positions = positions
        self.clubs = clubs
        self.num_players = len(expected_scores)

    def encode_player_indices(self, indices):
        decisions = np.zeros(self.num_players)
        decisions[indices] = 1
        return decisions

    def solve(self, current_team_indices, current_sub_indices, current_captain_indices, budget_now, sub_factor):
        current_team_decisions = self.encode_player_indices(current_team_indices)
        current_sub_decisions = self.encode_player_indices(current_sub_indices)
        current_captain_decisions = self.encode_player_indices(current_captain_indices)

        model = pulp.LpProblem("Transfer optimisation", pulp.LpMaximize)
        transfer_in_decisions_free = get_decision_array("transfer_in_free", self.num_players)
        transfer_in_decisions_paid = get_decision_array("transfer_in_paid", self.num_players)
        transfer_out_decisions = get_decision_array("transfer_out_paid", self.num_players)
        transfer_in_decisions = [a + b for a, b in zip(transfer_in_decisions_free, transfer_in_decisions_paid)]

        # only 1 free transfer
        model += sum(transfer_in_decisions_free) <= 1

        # points penalty for additional transfers
        transfer_penalty = sum(transfer_in_decisions_paid) * 4

        # budget constraint
        transfer_in_cost = sum(x * price for x, price in zip(transfer_in_decisions, self.buy_prices))
        transfer_out_cost = sum(x * price for x, price in zip(transfer_out_decisions, self.sell_prices))
        budget_next_week = budget_now + transfer_out_cost - transfer_in_cost
        model += budget_next_week >= 0

        # calculate new team from current team + transfers
        next_week_team = [
            current_team_decisions[i] + transfer_in_decisions[i] - transfer_out_decisions[i]
            for i in range(self.num_players)
        ]
        next_week_subs = current_sub_decisions
        next_week_captains = current_captain_decisions
        self.apply_formation_constraints(model, starters=next_week_team, subs=next_week_subs, captains=next_week_captains)

        # objective function:
        model += self.get_objective(next_week_team, next_week_subs, next_week_captains, sub_factor, transfer_penalty), "Objective"
        model.solve()

        return transfer_in_decisions, transfer_out_decisions

    def get_objective(self, starters, subs, captains, sub_factor, transfer_penalty):
        starter_points = sum(x * score for x, score in zip(starters, self.expected_scores))
        sub_points = sum(x * score for x, score in zip(subs, self.expected_scores))
        return starter_points + sub_factor * sub_points - transfer_penalty

    def apply_formation_constraints(self, model, starters, subs, captains):
        all_selected = [a + b for a, b in zip(starters, subs)]
        for position, data in position_data.items():
            # formation constraints
            model += sum(starter for starter, position in zip(starters, self.positions) if position == data["position_id"]) >= data["min_starters"]
            model += sum(starter for starter, position in zip(starters, self.positions) if position == data["position_id"]) <= data["max_starters"]
            model += sum(selected for selected, position in zip(all_selected, self.positions) if position == data["position_id"]) == data["num_total"]

        # club constraint
        for club_id in np.unique(self.clubs):
            model += sum(selected for selected, club in zip(all_selected, self.clubs) if club == club_id) <= 3  # max 3 players

        # total team size
        model += sum(all_selected) == 11
        model += sum(captains) == 1

        for i in range(self.num_players):
            model += (starters[i] - captains[i]) >= 0  # captain must also be on team
            model += (starters[i] + subs[i]) <= 1  # subs must not be on team


def optimise_transfers(current_team_indices, current_sub_indices, current_captain_indices, expected_scores, buy_prices,
                       sell_prices, positions, clubs, budget_now=0):
    num_players = len(expected_scores)

    current_team_decisions = np.zeros(num_players)
    current_team_decisions[current_team_indices] = 1

    current_sub_decisions = np.zeros(num_players)
    current_sub_decisions[current_sub_indices] = 1

    current_captain_decisions = np.zeros(num_players)
    current_captain_decisions[current_captain_indices] = 1

    model = pulp.LpProblem("Transfer optimisation", pulp.LpMaximize)
    transfer_in_decisions_free = get_decision_array("transfer_in_free", num_players)
    transfer_in_decisions_paid = get_decision_array("transfer_in_paid", num_players)
    transfer_out_decisions = get_decision_array("transfer_out_paid", num_players)
    transfer_in_decisions = [a + b for a, b in zip(transfer_in_decisions_free, transfer_in_decisions_paid)]

    model += sum(transfer_in_decisions_free) <= 1  # only 1 free transfer

    next_week_team = [
        current_team_decisions[i] + transfer_in_decisions[i] - transfer_out_decisions[i]
        for i in range(num_players)
    ]

    for i in range(num_players):
        # binary constraints on team selection
        model += next_week_team[i] <= 1
        model += next_week_team[i] >= 0
        # and on transfers
        model += (transfer_in_decisions[i] + transfer_out_decisions[i]) <= 1

    # budget constraint
    transfer_in_cost = sum(transfer_in_decisions[i] * buy_prices[i] for i in range(num_players))
    transfer_out_cost = sum(transfer_out_decisions[i] * sell_prices[i] for i in range(num_players))
    budget_next_week = budget_now + transfer_out_cost - transfer_in_cost
    model += budget_next_week >= 0

    # objective function:
    expt_points = sum(next_week_team[i] * expected_scores[i] for i in range(num_players))
    model += expt_points - sum(transfer_in_decisions_paid) * 4, "Objective"
    model.solve()

    for i in range(num_players):
        if transfer_in_decisions[i].value() == 1:
            print("Transferred in: {} {} {}".format(i, buy_prices[i], expected_scores[i]))
        if transfer_out_decisions[i].value() == 1:
            print("Transferred out: {} {} {}".format(i, sell_prices[i], expected_scores[i]))



def select_team(expected_scores, prices, positions, clubs, total_budget=100, sub_factor=0.2):
    num_players = len(expected_scores)
    model = pulp.LpProblem("Constrained value maximisation", pulp.LpMaximize)
    decisions = [
        pulp.LpVariable("x{}".format(i), lowBound=0, upBound=1, cat='Integer')
        for i in range(num_players)
    ]
    captain_decisions = [
        pulp.LpVariable("y{}".format(i), lowBound=0, upBound=1, cat='Integer')
        for i in range(num_players)
    ]
    sub_decisions = [
        pulp.LpVariable("z{}".format(i), lowBound=0, upBound=1, cat='Integer')
        for i in range(num_players)
    ]

    # objective function:
    model += sum((captain_decisions[i] + decisions[i] + sub_decisions[i] * sub_factor) * expected_scores[i]
                 for i in range(num_players)), "Objective"

    # cost constraint
    model += sum(
        (decisions[i] + sub_decisions[i]) * prices[i] for i in range(num_players)) <= total_budget  # total cost

    # position constraints
    # 1 starting goalkeeper
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 1) == 1
    # 2 total goalkeepers
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 1) == 2

    # 3-5 starting defenders
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 2) >= 3
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 2) <= 5
    # 5 total defenders
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 2) == 5

    # 3-5 starting midfielders
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 3) >= 3
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 3) <= 5
    # 5 total midfielders
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 3) == 5

    # 1-3 starting attackers
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 4) >= 1
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 4) <= 3
    # 3 total attackers
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 4) == 3

    # club constraint
    for club_id in np.unique(clubs):
        model += sum(
            decisions[i] + sub_decisions[i] for i in range(num_players) if clubs[i] == club_id) <= 3  # max 3 players

    model += sum(decisions) == 11  # total team size
    model += sum(captain_decisions) == 1  # 1 captain

    for i in range(num_players):
        model += (decisions[i] - captain_decisions[i]) >= 0  # captain must also be on team
        model += (decisions[i] + sub_decisions[i]) <= 1  # subs must not be on team

    model.solve()
    print("Total expected score = {}".format(model.objective.value()))

    return decisions, captain_decisions, sub_decisions