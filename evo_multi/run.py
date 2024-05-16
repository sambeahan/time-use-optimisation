import random

import objective_functions
import ngsa_ii

RUNS = 1

time_use_totals = [0, 0, 0]
outcome_totals = {}

last_time_use_totals = [0, 0, 0]
last_outcome_totals = {}

for i in range(RUNS):
    print("Run", i)
    ngsa_time_use = ngsa_ii.non_dominated_sorting_genetic_algorithm_II(
        population_size=50,
        mutation_rate=0.1,
        min_values=[4, 1, 0.5],
        max_values=[12, 18, 12],
        list_of_functions=[
            objective_functions.calc_stress,
            objective_functions.calc_hr,
            objective_functions.calc_sbp,
            objective_functions.calc_dbp,
            objective_functions.calc_bmi,
        ],
        generations=100,
        mu=1,
        eta=1,
    )

    competitors = [i for _ in range(len(ngsa_time_use))]
    random.shuffle(competitors)

    # number based dominance
    while len(competitors) > 1:
        next_round = []
        for i, solution1 in enumerate(competitors[::2]):
            if i + 1 < len(competitors):
                solution2 = competitors[i + 1]

                obj_vals1 = objective_functions.calc_outcomes(
                    ngsa_time_use[solution1][0],
                    ngsa_time_use[solution1][1],
                    ngsa_time_use[solution1][2],
                )
                obj_vals2 = objective_functions.calc_outcomes(
                    ngsa_time_use[solution2][0],
                    ngsa_time_use[solution2][1],
                    ngsa_time_use[solution2][2],
                )

                wins = 0

                for outcome, value in obj_vals1.items():
                    if value > obj_vals2[outcome]:
                        wins += 1

                if wins >= len(obj_vals1) / 2:
                    next_round.append(solution1)
                else:
                    next_round.append(solution2)

            else:
                next_round.append(solution1)

        competitors = next_round.copy()

    solution = ngsa_time_use[competitors[0]]
    # last_solution = ngsa_time_use[-1]

    time_use = solution[:3]
    # last_time_use = last_solution[:3]
    outcomes = objective_functions.calc_outcomes(solution[0], solution[1], solution[2])
    # last_outcomes = objective_functions.calc_outcomes(
    # last_solution[0], last_solution[1], last_solution[2]
    # )

    for outcome, value in outcomes.items():
        if outcome in outcome_totals:
            outcome_totals[outcome] += value
            # last_outcome_totals[outcome] += last_outcomes[outcome]
        else:
            outcome_totals[outcome] = value
            # last_outcome_totals[outcome] = last_outcomes[outcome]

    for j, time in enumerate(time_use):
        time_use_totals[j] += time
        # last_time_use_totals[j] += last_time_use[j]

print("\n----- AVERAGES -----")

for time in time_use_totals:
    print(time / RUNS)

for outcome in outcome_totals:
    print(outcome, outcome_totals[outcome] / RUNS)


# print("\nLasts:")
# for time in last_time_use_totals:
# print(time / RUNS)

# for outcome in last_outcome_totals:
# print(outcome, last_outcome_totals[outcome] / RUNS)
