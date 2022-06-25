from random import randint, uniform

from numpy.typing import NDArray
import numpy as np


def crossover(
    parent1: NDArray,
    parent2: NDArray,
    num_units: NDArray,
    item_values: NDArray
) -> tuple[NDArray, NDArray]:
    # Point at which we crossover should be random (can't be at the very ends though)
    cross_x = randint(0, parent1.shape[1])
    cross_y = randint(0, parent2.shape[0])
    crossover_point = (cross_x, cross_y)

    # One point crossover
    child1, child2 = make_offspring(parent1, parent2, (cross_x, cross_y))

    # Crossover algorithm used above can produce illegal solutions
    child1_values = np.dot(item_values, child1)
    child1 = repair(child1, crossover_point, num_units, child1_values)

    child2_values = np.dot(item_values, child2)
    child2 = repair(child2, crossover_point, num_units, child2_values)

    return child1, child2


def make_offspring(
    parent1: NDArray,
    parent2: NDArray,
    crossover_point: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Do one point crossover at the designated crossover point.

    Args:
        parent1 (np.ndarray): First parent solution to crossover
        parent2 (np.ndarray): Second parent solution to crossover
        cross (tuple[int, int]): x, y coordinates of point to split parents
    Return:
        np.ndarray: Child solution 1
        np.ndarray: Child solution 2
    """
    # Flat arrays make it easier to split and combine the parent arrays
    y, x = parent1.shape
    parent1_flat = parent1.reshape(x * y)
    parent2_flat = parent2.reshape(x * y)

    # Want to split on the flat array so we need to convert the crossover point
    cross = crossover_point[0] * x + crossover_point[1]

    # Crossover to make offspring
    offspring1_flat = np.append(parent1_flat[0:cross], parent2_flat[cross:])
    offspring2_flat = np.append(parent2_flat[0:cross], parent1_flat[cross:])

    # Return offspring with original 2D shape
    return (offspring1_flat.reshape(y, x), offspring2_flat.reshape(y, x))


def repair(
    solution: NDArray,
    crossover_point: tuple[int, int],
    num_units: NDArray,
    hamper_values: NDArray
) -> NDArray:
    """Fix illegal solutions.

    - Adds item to lowest value hamper
    - Removes item from highest value hamper

    Args:
        solution (NDArray): Chromosome that should be repaired
        crossover_point (tuple[int, int]): x, y coordinates of the crossover
            point that produced this chromosome
        num_units (NDArray): Number of units available for each item
    Return:
        NDArray: Repaired solution.
    """
    # Need to know so we can get the correct hamper to fix
    x, y = crossover_point
    to_fix = solution[x, :]

    # All units must be used no more no less
    expected_units = num_units[y]
    units = num_units.sum()
    diff = units - expected_units

    # There are additional items to be assigned
    if diff > 0:
        best_hampers = np.argsort(hamper_values)
        import pdb; pdb.set_trace()
    # More items have been assigned that are available
    elif diff < 0:
        hamper_value = np.dot(hamper_values, solution)
        best_hampers = np.argsort(hamper_value)[::-1]

    return solution


def add_missing_items():
    return


def remove_excess_items():
    return

