import random
import numpy as np
import time

import objective_functions
import ngsa_ii

RUNS = 100

time_use_totals = [0, 0, 0]
outcome_totals = {}

health_vals = {
    "Stress Level": [],
    "Resting Heart Rate": [],
    "Systolic Blood Pressure": [],
    "Diastolic Blood Pressure": [],
    "BMI": [],
}

time_use_vals = [[], [], []]

start = time.time()

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

    competitors = [i for i in range(len(ngsa_time_use))]
    random.shuffle(competitors)

    # number based dominance
    while len(competitors) > 1:
        next_round = []
        for i in range(0, len(competitors), 2):
            # print(i)
            solution1 = competitors[i]
            if i + 1 < len(competitors):
                # print(i + 1)
                solution2 = competitors[i + 1]

                obj_vals1 = [
                    objective_functions.calc_stress(ngsa_time_use[solution1]),
                    objective_functions.calc_hr(ngsa_time_use[solution1]),
                    objective_functions.calc_sbp(ngsa_time_use[solution1]),
                    objective_functions.calc_dbp(ngsa_time_use[solution1]),
                    objective_functions.calc_bmi(ngsa_time_use[solution1]),
                ]

                # print("a:", obj_vals1)

                obj_vals2 = [
                    objective_functions.calc_stress(ngsa_time_use[solution2]),
                    objective_functions.calc_hr(ngsa_time_use[solution2]),
                    objective_functions.calc_sbp(ngsa_time_use[solution2]),
                    objective_functions.calc_dbp(ngsa_time_use[solution2]),
                    objective_functions.calc_bmi(ngsa_time_use[solution2]),
                ]

                # print("b:", obj_vals2)

                wins = 0

                for i, outcome in enumerate(obj_vals1):
                    if outcome < obj_vals2[i]:
                        wins += 1

                if wins >= len(obj_vals1) / 2:
                    next_round.append(solution1)
                    # print("Winner: a")
                else:
                    next_round.append(solution2)
                    # print("Winner: b")

            else:
                next_round.append(solution1)

        competitors = next_round.copy()

        # print()

    solution = ngsa_time_use[competitors[0]]

    time_use = solution[:3]
    outcomes = objective_functions.calc_outcomes(solution[0], solution[1], solution[2])
    print(time_use)
    print(outcomes)

    for outcome, value in outcomes.items():
        if outcome in outcome_totals:
            outcome_totals[outcome] += value
        else:
            outcome_totals[outcome] = value

        health_vals[outcome].append(value)

    for j, time_spent in enumerate(time_use):
        time_use_totals[j] += time_spent
        time_use_vals[j].append(time_spent)

end = time.time()

print("\n----- AVERAGES -----")

for time_spent in time_use_totals:
    print(time_spent / RUNS)

for outcome, value in outcome_totals.items():
    print(outcome, value / RUNS)


print("\nStandard deviation:")
for activity_time in time_use_vals:
    print(np.std(activity_time))

for outcome, values in health_vals.items():
    values = np.array(values)
    print(outcome + ":", np.std(values))


print("\nRuntime:", (end - start) / RUNS)
