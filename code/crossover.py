from random import randint, uniform

from numpy.typing import NDArray
import numpy as np


def crossover(
    parents: list[NDArray],
    num_offspring: int,
    num_units: NDArray,
    item_values: NDArray,
    crossover_rate: float,
) -> list:
    # Iterate through parents
    children = []
    for i in range(0, num_offspring, 2):
        # Get pair of parents
        parent1 = parents[i]
        parent2 = parents[i + 1]

        if uniform(0, 1) > crossover_rate:
            children.append(parent1)
            children.append(parent2)
            continue

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

        children.append(child1)
        children.append(child2)

    return children


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
    to_fix = solution[y, :]

    # All units must be used no more no less
    expected_units = num_units[y]
    units = to_fix.sum()
    diff = units - expected_units

    print(units)
    print(diff)

    # There are additional items to be assigned
    if diff > 0:
        solution[y, :] = remove_excess_items(to_fix, hamper_values, diff)
    # More items have been assigned that are available
    elif diff < 0:
        solution[y, :] = add_missing_items(to_fix, hamper_values, diff)
        pass

    return solution


def add_missing_items(
    to_fix: NDArray,
    hamper_values: NDArray,
    num_to_add: int,
) -> NDArray:
    # Indexes of hamper values in order of cheapest to most expensive
    cheapest_hampers = np.argsort(hamper_values)

    # Get binary hamper assignments in order of hamper value
    sorted_assignment = to_fix[cheapest_hampers]

    # Get indices of hampers that are currently 0 so we can set them to 1
    cheapest_zeros = cheapest_hampers[sorted_assignment==0]
    to_flip = cheapest_zeros[0:num_to_add]

    # Set the cheapest hampers that are current 0 to 1
    to_fix[to_flip] = 1

    return to_fix


def remove_excess_items(
    to_fix: NDArray,
    hamper_values: NDArray,
    num_to_remove: int,
) -> NDArray:
    # Indexes of hamper values in order of cheapest to most expensive
    cheapest_hampers = np.argsort(hamper_values)[::-1]

    # Get chromosome slice in order of cheapest to most expensive hamper
    sorted_assignment = to_fix[cheapest_hampers]

    # Get indices of hampers that are currently 1 so we can set them to 0
    cheapest_zeros = cheapest_hampers[sorted_assignment==1]
    to_flip = cheapest_zeros[0:num_to_remove]

    # Set the cheapest hampers that are current 0 to 1
    to_fix[to_flip] = 0

    return to_fix

