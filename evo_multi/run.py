import objective_functions
import ngsa_ii
import numpy as np

RUNS = 100

time_use_totals = [0, 0, 0]
outcome_totals = {}

for i in range(RUNS):
    print("Run", i)
    ngsa_time_use = ngsa_ii.non_dominated_sorting_genetic_algorithm_II(
        population_size=50,
        mutation_rate=0.1,
        min_values=[4, 2, 1],
        max_values=[12, 18, 8],
        list_of_functions=[
            objective_functions.calc_stress,
            objective_functions.calc_hr,
            objective_functions.calc_sbp,
            objective_functions.calc_dbp,
            objective_functions.calc_bmi,
        ],
        generations=50,
        mu=1,
        eta=1,
    )

    solution = ngsa_time_use[0]

    time_use = solution[:3]
    outcomes = objective_functions.calc_outcomes(solution[0], solution[1], solution[2])

    print()
    print(time_use)
    print(outcomes)

    for outcome, value in outcomes.items():
        if outcome in outcome_totals:
            outcome_totals[outcome] += value
        else:
            outcome_totals[outcome] = value

    for j, time in enumerate(time_use):
        time_use_totals[j] += time

print("\n----- AVERAGES -----")

for time in time_use_totals:
    print(time / RUNS)

for outcome in outcome_totals:
    print(outcome, outcome_totals[outcome] / RUNS)
