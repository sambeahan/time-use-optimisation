import objective_functions
import ngsa_ii
import numpy as np

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

time_use_totals = [0, 0, 0]
outcome_totals = {}
num_solns = np.shape(ngsa_time_use)[0]

for i, solution in enumerate(ngsa_time_use):
    time_use = solution[:3]
    outcomes = objective_functions.calc_outcomes(solution[0], solution[1], solution[2])

    if i < 3:
        print()
        print(time_use)
        print(outcomes)

    for outcome in outcomes:
        if outcome in outcome_totals:
            outcome_totals[outcome] += outcomes[outcome]
        else:
            outcome_totals[outcome] = outcomes[outcome]

    for i, time in enumerate(time_use):
        time_use_totals[i] += time

print("\n----- AVERAGES -----")

for time in time_use_totals:
    print(time / num_solns)

for outcome in outcome_totals:
    print(outcome, outcome_totals[outcome] / num_solns)
