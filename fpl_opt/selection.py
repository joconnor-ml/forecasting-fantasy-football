import numpy as np
import pandas as pd
import pulp

COLS_TO_PRINT = [
    "first_name",
    "second_name",
    "expected_score",
    "price",
    "first_team",
    "captain",
    "sub",
]


def select_team(
    expected_scores,
    prices,
    positions,
    clubs,
    total_budget=100,
    sub_factors=0.2,
    playing_chance=None,
):
    if type(sub_factors) in (float, int):
        sub_factors = [sub_factors] * 4
    if playing_chance is None:
        playing_chance = np.ones_like(expected_scores)
    num_players = len(expected_scores)
    model = pulp.LpProblem("Constrained value maximisation", pulp.LpMaximize)
    decisions = [
        pulp.LpVariable(f"first_team_{i}", lowBound=0, upBound=1, cat="Integer")
        for i in range(num_players)
    ]
    captain_decisions = [
        pulp.LpVariable(f"captain_{i}", lowBound=0, upBound=1, cat="Integer")
        for i in range(num_players)
    ]
    NUM_SUBS = 4
    sub_decisions = [
        [
            pulp.LpVariable(f"sub{nsub}_{i}", lowBound=0, upBound=1, cat="Integer")
            for i in range(num_players)
        ]
        for nsub in range(NUM_SUBS)
    ]
    total_sub_decisions = [
        sum(sub_decisions[j][i] for j in range(NUM_SUBS)) for i in range(num_players)
    ]

    # objective function:
    model += (
        sum(
            (
                captain_decisions[i]
                + decisions[i] * playing_chance[i]
                + sum(
                    sub_decisions[j][i] * playing_chance[i] * sub_factors[j]
                    for j in range(NUM_SUBS)
                )
            )
            * expected_scores[i]
            for i in range(num_players)
        ),
        "Objective",
    )

    # cost constraint
    model += (
        sum(
            (decisions[i] + total_sub_decisions[i]) * prices[i]
            for i in range(num_players)
        )
        <= total_budget
    )  # total cost

    # position constraints
    # 1 starting goalkeeper
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 1) == 1
    # 2 total goalkeepers
    model += (
        sum(
            decisions[i] + total_sub_decisions[i]
            for i in range(num_players)
            if positions[i] == 1
        )
        == 2
    )

    # 3-5 starting defenders
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 2) >= 3
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 2) <= 5
    # 5 total defenders
    model += (
        sum(
            decisions[i] + total_sub_decisions[i]
            for i in range(num_players)
            if positions[i] == 2
        )
        == 5
    )

    # 3-5 starting midfielders
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 3) >= 3
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 3) <= 5
    # 5 total midfielders
    model += (
        sum(
            decisions[i] + total_sub_decisions[i]
            for i in range(num_players)
            if positions[i] == 3
        )
        == 5
    )

    # 1-3 starting attackers
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 4) >= 1
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 4) <= 3
    # 3 total attackers
    model += (
        sum(
            decisions[i] + total_sub_decisions[i]
            for i in range(num_players)
            if positions[i] == 4
        )
        == 3
    )

    # force GK into sub slot 4
    model += (
        sum(sub_decisions[3][i] for i in range(num_players) if positions[i] == 1) == 1
    )

    # club constraint
    for club_id in np.unique(clubs):
        model += (
            sum(
                decisions[i] + total_sub_decisions[i]
                for i in range(num_players)
                if clubs[i] == club_id
            )
            <= 3
        )  # max 3 players

    model += sum(decisions) == 11  # total team size
    model += sum(captain_decisions) == 1  # 1 captain

    for i in range(num_players):
        model += (
            decisions[i] - captain_decisions[i]
        ) >= 0  # captain must also be on team
        model += (
            decisions[i] + total_sub_decisions[i]
        ) <= 1  # subs must not be on team

    model.solve()
    print(f"Total expected score = {model.objective.value()}")

    return decisions, captain_decisions, total_sub_decisions


def get_selection_df(decisions, captain_decisions, sub_decisions, player_df):
    selection_data = []
    for i in range(len(decisions)):
        if decisions[i].value() == 1:
            selection_data.append(
                {
                    "player_index": i,
                    "first_team": True,
                    "sub": False,
                    "captain": bool(captain_decisions[i].value()),
                }
            )

    for i in range(len(sub_decisions)):
        if sub_decisions[i].value() == 1:
            selection_data.append(
                {"player_index": i, "first_team": False, "sub": True, "captain": False}
            )

    return (
        pd.DataFrame(selection_data)
        .set_index("player_index")
        .join(player_df, how="left")
    )
