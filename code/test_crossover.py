import numpy as np
import random
import crossover


def test_make_offspring():
    random.seed(0)

    parent1 = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]])
    parent2 = np.array([[0, 1, 1, 0], [1, 0, 1, 0], [0, 0, 1, 1]])

    result1 = np.array([[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 1, 1]])
    result2 = np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0]])

    results = crossover.make_offspring(parent1, parent2)
    np.testing.assert_array_equal(results[0], result1)
    np.testing.assert_array_equal(results[1], result2)

