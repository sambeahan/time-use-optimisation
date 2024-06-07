"""Single objective genetic algorithm for continuous function optimization"""

# Code from https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/
# Adapted with stress objective function, and normalising solution to 24 hour total

import sys
import os
from numpy.random import randint
from numpy.random import rand

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import objective_functions


OBJECTIVE = "stress"


def objective(sleep_time, active_time, sedentary_time):
    """The objective function for stress level"""
    return objective_functions.calc_stress([sleep_time, active_time, sedentary_time])


def decode(bounds, n_bits, bitstring):
    """Decoding bitstring to obtain values"""
    decoded = list()
    largest = 2**n_bits
    for i in range(len(bounds)):
        # extract the substring
        start, end = i * n_bits, (i * n_bits) + n_bits
        substring = bitstring[start:end]
        # convert bitstring to a string of chars
        chars = "".join([str(s) for s in substring])
        # convert string to integer
        integer = int(chars, 2)
        # scale integer to desired range
        value = bounds[i][0] + (integer / largest) * (bounds[i][1] - bounds[i][0])
        # store
        decoded.append(value)

    total = sum(decoded)
    for i, val in enumerate(decoded):
        decoded[i] = (decoded[i] * 24) / total
    return decoded


def selection(pop, scores, k=3):
    """Tournament selection"""
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k - 1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


def crossover(p1, p2, r_cross):
    """Crossover two parents to create two children"""
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = randint(1, len(p1) - 2)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]


def mutation(bitstring, r_mut):
    """Mutates bitstring at mutation rate"""
    for i in range(len(bitstring)):
        # check for a mutation
        if rand() < r_mut:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]


def genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut):
    # initial population of random bitstring
    pop = [randint(0, 2, n_bits * len(bounds)).tolist() for _ in range(n_pop)]
    # keep track of best solution
    first_sol_decoded = decode(bounds, n_bits, pop[0])
    best, best_eval = 0, objective(
        first_sol_decoded[0], first_sol_decoded[1], first_sol_decoded[2]
    )
    # enumerate generations
    for gen in range(n_iter):
        # decode population
        decoded = [decode(bounds, n_bits, p) for p in pop]
        # evaluate all candidates in the population
        scores = [objective(d[0], d[1], d[2]) for d in decoded]
        # check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                print(">%d, new best f(%s) = %f" % (gen, decoded[i], scores[i]))
        # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i + 1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation(c, r_mut)
                # store for next generation
                children.append(c)
        # replace population
        pop = children
    return [best, best_eval]


if __name__ == "__main__":
    # define variable bounds
    bounds = [[0, 24.0], [0, 24.0], [0, 24.0]]

    # define the total iterations
    iteration_count = 100

    # bits per variable
    n_bits = 16

    # define the population size
    population_size = 100

    # crossover rate
    crossover_rate = 0.9

    # mutation rate
    mutation_rate = 1.0 / (float(n_bits) * len(bounds))

    best_solution, score = genetic_algorithm(
        objective,
        bounds,
        n_bits,
        iteration_count,
        population_size,
        crossover_rate,
        mutation_rate,
    )
    print("Done!")
    decoded = decode(bounds, n_bits, best_solution)
    print("f(%s) = %f" % (decoded, score))
