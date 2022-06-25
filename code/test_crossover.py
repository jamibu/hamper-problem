import numpy as np
import random
import crossover


def test_make_offspring():
    # Parent solutions
    parent1 = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]])
    parent2 = np.array([[0, 1, 1, 0], [1, 0, 1, 0], [0, 0, 1, 1]])
    # Expected child solutions
    result1 = np.array([[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 1, 1]])
    result2 = np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0]])
    co_point = (1, 3)

    results = crossover.make_offspring(parent1, parent2, co_point)
    np.testing.assert_array_equal(results[0], result1)
    np.testing.assert_array_equal(results[1], result2)


def test_crossover():
    random.seed(0)


def test_repair():
    # Inputs
    solution = np.array([[1, 0, 1, 0], [0, 1, 1, 1], [1, 1, 0, 0]])
    crossover_point = (3, 1)
    num_units = np.array([2, 2, 2])
    hamper_values = np.array([5, 4, 3, 2])
    # Expected repaired solution
    expected = np.array([[1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 0]])
    # Call the function
    result = crossover.repair(solution, crossover_point, num_units, hamper_values)
    np.testing.assert_array_equal(result, expected)

    # Inputs
    solution = np.array([[1, 0, 1, 0], [0, 1, 0, 0], [1, 1, 0, 0]])
    crossover_point = (3, 1)
    num_units = np.array([2, 2, 2])
    hamper_values = np.array([5, 4, 3, 2])
    # Expected repaired solution
    expected = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]])
    # Call the function
    result = crossover.repair(solution, crossover_point, num_units, hamper_values)


def test_add_missing_items():
    to_fix = np.array([0, 1, 0, 0])
    hamper_values = np.array([5, 5, 5, 4])
    num_to_add = 1

    expected = np.array([0, 1, 0, 1])
    result = crossover.add_missing_items(to_fix, hamper_values, num_to_add)

    np.testing.assert_array_equal(expected, result)


def test_remove_excess_items():
    to_fix = np.array([0, 1, 1, 1])
    hamper_values = np.array([5, 5, 6, 4])
    num_to_remove = 1

    expected = np.array([0, 1, 0, 1])
    result = crossover.remove_excess_items(to_fix, hamper_values, num_to_remove)

    np.testing.assert_array_equal(expected, result)

