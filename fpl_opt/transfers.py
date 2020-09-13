import pulp
import numpy as np

position_data = {
    "gk": {"position_id": 1, "min_starters": 1, "max_starters": 1, "num_total": 2},
    "df": {"position_id": 2, "min_starters": 3, "max_starters": 5, "num_total": 5},
    "mf": {"position_id": 3, "min_starters": 3, "max_starters": 5, "num_total": 5},
    "fw": {"position_id": 4, "min_starters": 1, "max_starters": 3, "num_total": 3},
}


def get_decision_array(name, length):
    return np.array([
        pulp.LpVariable("{}_{}".format(name, i), lowBound=0, upBound=1, cat='Integer')
        for i in range(length)
    ])


class TransferOptimiser:
    def __init__(self, expected_scores, buy_prices, sell_prices, positions, clubs):
        self.expected_scores = expected_scores
        self.buy_prices = buy_prices
        self.sell_prices = sell_prices
        self.positions = positions
        self.clubs = clubs
        self.num_players = len(buy_prices)

    def instantiate_decision_arrays(self):
        # we will make transfers in and out of the squad, and then pick subs and captains from that squad
        transfer_in_decisions_free = get_decision_array("transfer_in_free", self.num_players)
        transfer_in_decisions_paid = get_decision_array("transfer_in_paid", self.num_players)
        transfer_out_decisions = get_decision_array("transfer_out_paid", self.num_players)
        # total transfers in will be useful later
        transfer_in_decisions = transfer_in_decisions_free + transfer_in_decisions_paid

        sub_decisions = get_decision_array("subs", self.num_players)
        captain_decisions = get_decision_array("captain", self.num_players)
        return transfer_in_decisions_free, transfer_in_decisions_paid, transfer_out_decisions, transfer_in_decisions, sub_decisions, captain_decisions

    def encode_player_indices(self, indices):
        decisions = np.zeros(self.num_players)
        decisions[indices] = 1
        return decisions

    def apply_transfer_constraints(self, model, transfer_in_decisions_free, transfer_in_decisions,
                                   transfer_out_decisions, budget_now):
        # only 1 free transfer
        model += sum(transfer_in_decisions_free) <= 1

        # budget constraint
        transfer_in_cost = sum(transfer_in_decisions * self.buy_prices)
        transfer_out_cost = sum(transfer_out_decisions * self.sell_prices)
        budget_next_week = budget_now + transfer_out_cost - transfer_in_cost
        model += budget_next_week >= 0


    def solve(self, current_squad_indices, budget_now, sub_factor):
        current_squad_decisions = self.encode_player_indices(current_squad_indices)

        model = pulp.LpProblem("Transfer optimisation", pulp.LpMaximize)
        transfer_in_decisions_free, transfer_in_decisions_paid, transfer_out_decisions, transfer_in_decisions, sub_decisions, captain_decisions = self.instantiate_decision_arrays()

        # calculate new team from current team + transfers
        next_week_squad = current_squad_decisions + transfer_in_decisions - transfer_out_decisions
        starters = next_week_squad - sub_decisions

        # points penalty for additional transfers
        transfer_penalty = sum(transfer_in_decisions_paid) * 4

        self.apply_transfer_constraints(model, transfer_in_decisions_free, transfer_in_decisions,
                                        transfer_out_decisions, budget_now)
        self.apply_formation_constraints(model, squad=next_week_squad, starters=starters,
                                         subs=sub_decisions, captains=captain_decisions)

        # objective function:
        model += self.get_objective(starters, sub_decisions, captain_decisions, sub_factor, transfer_penalty, self.expected_scores), "Objective"
        status = model.solve()

        print("Solver status: {}".format(status))

        return transfer_in_decisions, transfer_out_decisions, starters, sub_decisions, captain_decisions

    def get_objective(self, starters, subs, captains, sub_factor, transfer_penalty, scores):
        starter_points = sum(starters * scores)
        sub_points = sum(subs * scores) * sub_factor
        captain_points = sum(captains * scores)
        return starter_points + sub_points + captain_points - transfer_penalty

    def apply_formation_constraints(self, model, squad, starters, subs, captains):
        for position, data in position_data.items():
            # formation constraints
            model += sum(starter for starter, position in zip(starters, self.positions) if position == data["position_id"]) >= data["min_starters"]
            model += sum(starter for starter, position in zip(starters, self.positions) if position == data["position_id"]) <= data["max_starters"]
            model += sum(selected for selected, position in zip(squad, self.positions) if position == data["position_id"]) == data["num_total"]

        # club constraint
        for club_id in np.unique(self.clubs):
            model += sum(selected for selected, club in zip(squad, self.clubs) if club == club_id) <= 3  # max 3 players

        # total team size
        model += sum(starters) == 11
        model += sum(squad) == 15
        model += sum(captains) == 1

        for i in range(self.num_players):
            model += (starters[i] - captains[i]) >= 0  # captain must also be on team
            model += (starters[i] + subs[i]) <= 1  # subs must not be on team


def get_decision_array_2d(name, n_players, n_weeks):
    return np.array([[
        pulp.LpVariable("{}_{}_w{}".format(name, i, j), lowBound=0, upBound=1, cat='Integer')
        for i in range(n_players)
    ] for j in range(n_weeks)])


class MultiHorizonTransferOptimiser(TransferOptimiser):
    """We now plan transfer decisions over multiple weeks. This means we need a 2d array of expected
    scores (n_players x n_weeks) and 2d arrays of decision variables"""
    def __init__(self, expected_scores, buy_prices, sell_prices, positions, clubs,
                 n_weeks):
        super().__init__(expected_scores, buy_prices, sell_prices, positions, clubs)
        self.num_weeks = n_weeks

    def instantiate_decision_arrays(self):
        # we will make transfers in and out of the squad, and then pick subs and captains from that squad
        transfer_in_decisions_free = get_decision_array_2d("transfer_in_free", self.num_players, self.num_weeks)
        transfer_in_decisions_paid = get_decision_array_2d("transfer_in_paid", self.num_players, self.num_weeks)
        transfer_out_decisions = get_decision_array_2d("transfer_out_paid", self.num_players, self.num_weeks)
        # total transfers in will be useful later
        transfer_in_decisions = [a + b for a, b in zip(transfer_in_decisions_free, transfer_in_decisions_paid)]

        sub_decisions = get_decision_array_2d("subs", self.num_players, self.num_weeks)
        captain_decisions = get_decision_array_2d("captain", self.num_players, self.num_weeks)
        return transfer_in_decisions_free, transfer_in_decisions_paid, transfer_out_decisions, transfer_in_decisions, sub_decisions, captain_decisions

    def solve(self, current_squad_indices, budget_now, sub_factor):
        current_squad_decisions = self.encode_player_indices(current_squad_indices)
        model = pulp.LpProblem("Transfer optimisation", pulp.LpMaximize)
        (transfer_in_decisions_free_all, transfer_in_decisions_paid_all, transfer_out_decisions_all,
         transfer_in_decisions_all, sub_decisions_all, captain_decisions_all) = self.instantiate_decision_arrays()

        total_points = 0
        for w in range(self.num_weeks):
            transfer_in_decisions_free = transfer_in_decisions_free_all[w]
            transfer_in_decisions_paid = transfer_in_decisions_paid_all[w]
            transfer_out_decisions = transfer_out_decisions_all[w]
            transfer_in_decisions = transfer_in_decisions_all[w]
            sub_decisions = sub_decisions_all[w]
            captain_decisions = captain_decisions_all[w]

            # calculate new team from current team + transfers
            next_week_squad = current_squad_decisions + transfer_in_decisions - transfer_out_decisions
            starters = next_week_squad - sub_decisions

            # points penalty for additional transfers
            transfer_penalty = sum(transfer_in_decisions_paid) * 4

            self.apply_transfer_constraints(model, transfer_in_decisions_free, transfer_in_decisions,
                                            transfer_out_decisions, budget_now)
            self.apply_formation_constraints(model, squad=next_week_squad, starters=starters,
                                             subs=sub_decisions, captains=captain_decisions)

            # objective function:
            total_points += self.get_objective(starters, sub_decisions, captain_decisions, sub_factor, transfer_penalty, self.expected_scores[w])
            print(type(total_points))
            current_squad_decisions = next_week_squad

        model += total_points, "Objective"
        model.solve()

        return transfer_in_decisions_all, transfer_out_decisions_all, sub_decisions_all, sub_decisions_all, captain_decisions_all
