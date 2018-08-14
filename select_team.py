"""Linear optimisation routines to select teams from expected points and prices."""
import pulp


def select_players(df, points_col="joe_ep", total_cost=83, sub_factor=0.5):
    model = pulp.LpProblem("Constrained expected score maximisation", pulp.LpMaximize)
    # variables are binary "buy/don't buy" for each player
    starters = [pulp.LpVariable("x{}".format(i), lowBound=0, upBound=1, cat='Integer')
                for i in df.index]
    subs = [pulp.LpVariable("y{}".format(i), lowBound=0, upBound=1, cat='Integer')
            for i in df.index]
    # Objective
    model += sum(df[points_col].fillna(0).iloc[i] * starters[i]
                 for i in range(df.shape[0])) + sum(sub_factor * df[points_col].fillna(0).iloc[i] * subs[i]
                                                    for i in range(df.shape[0])), "Expected Points"

    # exactly 1 starting keeper
    model += sum(v for i, v in enumerate(starters) if df["element_type"].iloc[i] == 1) == 1
    # 2 keepers total
    model += sum(v for i, v in enumerate(starters) if df["element_type"].iloc[i] == 1) + sum(
        v for i, v in enumerate(subs) if df["element_type"].iloc[i] == 1) == 2

    # at least 3 starting defenders
    model += sum(v for i, v in enumerate(starters) if df["element_type"].iloc[i] == 2) >= 3
    # 5 defenders total
    model += sum(v for i, v in enumerate(starters) if df["element_type"].iloc[i] == 2) + sum(
        v for i, v in enumerate(subs) if df["element_type"].iloc[i] == 2) == 5

    # at least 3 starting midfielders
    model += sum(v for i, v in enumerate(starters) if df["element_type"].iloc[i] == 3) >= 3
    # 5 midfielders total
    model += sum(v for i, v in enumerate(starters) if df["element_type"].iloc[i] == 3) + sum(
        v for i, v in enumerate(subs) if df["element_type"].iloc[i] == 3) == 5

    # at least 1 starting attacker
    model += sum(v for i, v in enumerate(starters) if df["element_type"].iloc[i] == 4) >= 1
    # 3 attackers total
    model += sum(v for i, v in enumerate(starters) if df["element_type"].iloc[i] == 4) + sum(
        v for i, v in enumerate(subs) if df["element_type"].iloc[i] == 4) == 3

    # cost constraint
    model += sum(v for i, v in enumerate(starters)) == 11

    # players can't be in lineup *and* on bench
    for i in range(df.shape[0]):
        model += starters[i] + subs[i] <= 1

    # cost constraint
    model += sum(df["now_cost"].iloc[i] * starters[i] for i in range(df.shape[0])) + \
             sum(df["now_cost"].iloc[i] * subs[i] for i in range(df.shape[0])) \
             <= total_cost * 10

    model.solve()

    selected_starters = [i for i, v in enumerate(starters) if v.varValue > 0]
    selected_subs = [i for i, v in enumerate(subs) if v.varValue > 0]

    return df.iloc[selected_starters].sort_values("element_type"), \
           df.iloc[selected_subs].sort_values("element_type")
