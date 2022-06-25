import numpy as np
from selection import fitness_calc, selection
from initialise import initialise_population


def test_fitness_calc():

    chromosome = np.array([
        [1, 1, 0, 0],
        [1, 0, 1, 0],
    ])
    item_values = np.array([10, 5])
    result = fitness_calc(chromosome, item_values, 7)
    assert result == 20


def test_selection():
    pop = initialise_population(10, [1, 2, 3, 4, 5], 20)
    fitness = np.array([8, 2, 7, 5, 1, 6, 0, 9, 4, 3])
    expected = [pop[6], pop[4], pop[1], pop[9]]
    result = selection(fitness, 4, pop)

    np.testing.assert_array_equal(np.array(expected), np.array(result))

