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

    results = crossover.onepoint_crossover(parent1, parent2, co_point)
    np.testing.assert_array_equal(results[0], result1)
    np.testing.assert_array_equal(results[1], result2)


# TODO add tests for cross_and_repair


def test_crossover():
    random.seed(0)

    population = [
        np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]]),
        np.array([[0, 1, 1, 0], [1, 0, 0, 1], [0, 0, 1, 1]]),
        np.array([[0, 1, 1, 0], [1, 1, 0, 0], [0, 1, 1, 0]]),
        np.array([[1, 0, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]]),
    ]
    num_units = np.array([2, 2, 2])
    item_values = np.array([3, 2, 1])

    # Arrays before repair.
    # [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1] - No repair needed
    # [0, 1, 1, 0], [1, 0, 0, 1], [1, 1, 0, 0] - No repair needed
    # [0, 1, 1, 0], [1, 1, 0, 1], [1, 0, 0, 1] - Values: 3, 5, 3, 3 Remove [1, 1]
    # [1, 0, 1, 0], [0, 0, 1, 0], [0, 1, 1, 0] - Values: 3, 1, 5, 0 Add [1, 3]

    expected = [
        np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]]),
        np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 1, 0, 0]]),
        np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1]]),
        np.array([[1, 0, 1, 0], [0, 0, 1, 1], [0, 1, 1, 0]]),
    ]
    result = crossover.crossover(population, 4, num_units, item_values)

    print(expected)
    print(result)

    np.testing.assert_array_equal(result, expected)


def test_repair():
    # Inputs
    solution = np.array([[1, 0, 1, 0], [0, 1, 1, 1], [1, 1, 0, 0]])
    crossover_point = (1, 3)
    num_units = np.array([2, 2, 2])
    hamper_values = np.array([5, 4, 3, 2])
    # Expected repaired solution
    expected = np.array([[1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 0]])
    # Call the function
    result = crossover.repair(solution, crossover_point, num_units, hamper_values)
    np.testing.assert_array_equal(result, expected)

    # Inputs
    solution = np.array([[1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 0, 0]])
    crossover_point = (1, 3)
    num_units = np.array([2, 2, 2])
    hamper_values = np.array([5, 4, 3, 2])
    # Expected repaired solution
    expected = np.array([[1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 0]])
    # Call the function
    result = crossover.repair(solution, crossover_point, num_units, hamper_values)

    np.testing.assert_array_equal(result, expected)


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

