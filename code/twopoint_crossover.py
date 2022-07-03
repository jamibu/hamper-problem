from random import randint

from numpy.typing import NDArray
from numpy.random import default_rng
import numpy as np


CROSSPOINT = tuple[int, int]


def crossover(
    parents: list[NDArray],
    num_offspring: int,
    num_units: NDArray
) -> list:
    # Iterate through parents
    children = []
    for i in range(0, num_offspring - 1, 2):
        # Get pair of parents
        parent1 = parents[i]
        parent2 = parents[i + 1]

        child1, child2 = cross_and_repair(
            parent1,
            parent2,
            num_units
        )

        children.append(child1)
        children.append(child2)

    return children


def cross_and_repair(parent1, parent2, num_units):
    # Point at which we crossover should be random (can't be at the very ends though)
    cross1_x = randint(0, parent1.shape[1] - 3)
    cross1_y = randint(0, parent2.shape[0] - 3)

    cross2_x = randint(cross1_x, parent1.shape[1] - 1)
    cross2_y = randint(cross1_y, parent2.shape[0] - 1)

    crossover_points = ((cross1_x, cross1_y), (cross2_x, cross2_y))
    # One point crossover
    child1, child2 = twopoint_crossover(
        parent1,
        parent2,
        crossover_points
    )

    # Crossover algorithm used above can produce illegal solutions
    repaired1 = repair(child1, num_units)
    repaired2 = repair(child2, num_units)

    return repaired1, repaired2


def twopoint_crossover(
    parent1: NDArray,
    parent2: NDArray,
    crossover_points: tuple[CROSSPOINT, CROSSPOINT],
) -> tuple[np.ndarray, np.ndarray]:
    """Do two point crossover at the designated crossover point.

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
    cross1 = crossover_points[0][0] * x + crossover_points[0][1]
    cross2 = crossover_points[1][0] * x + crossover_points[1][1]

    # Crossover to make offspring
    offspring1_flat = np.append(
        parent1_flat[0:cross1],
        parent2_flat[cross1:cross2],
    )
    offspring1_flat = np.append(
        offspring1_flat,
        parent1_flat[cross2:],
    )

    offspring2_flat = np.append(
        parent2_flat[0:cross1],
        parent1_flat[cross1:cross2],
    )
    offspring2_flat = np.append(
        offspring2_flat,
        parent2_flat[cross2:],
    )

    # Return offspring with original 2D shape
    return (offspring1_flat.reshape(y, x), offspring2_flat.reshape(y, x))


def repair(
    solution: NDArray,
    num_units: NDArray,
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
    rng = default_rng()
    diffs = solution.sum(axis=1) - num_units

    too_high = np.argwhere(diffs > 0)
    too_low = np.argwhere(diffs < 0)

    for i in too_high.flatten():
        ones =  np.argwhere(solution[i, :] == 1)
        ones = ones.flatten()
        to_change = rng.choice(ones, size=diffs[i], replace=False)
        solution[i, to_change] = 0

    for i in too_low.flatten():
        zeroes =  np.argwhere(solution[i, :] == 0)
        zeroes = zeroes.flatten()
        to_change = rng.choice(zeroes, size=np.abs(diffs[i]), replace=False)
        solution[i, to_change] = 1

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

