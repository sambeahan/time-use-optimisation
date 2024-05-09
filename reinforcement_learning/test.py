from objective_functions import calc_outcomes


results = calc_outcomes(7.8, 14.2, 2)

for result, value in results.items():
    print(result, ":", value)
