import numpy as np


def randomly_distribute_item(num_hampers: int, num_units: int) -> np.ndarray:
    """Randomly create array representing the assignments of a single item.

    - This is a single row in the chromosome
    - Each value in array represents a hamper
    - Values are binary (1 if the items is in the hamper 0 if not)

    Args:
        num_hampers (int): Number of hampers that will be made.
        num_units (int): Number of units that are available for an item
    Returns:
        np.ndarray: Array of binary values with length of num_hampers + num_units
            containing num_units of 1s
    """
    # All hampers that don't have the item should be zero
    num_zeros = num_hampers - num_units

    # Want the number of zeros in the hamper to be equal to the number of units
    # of an item
    item_arr = np.array([0] * num_zeros + [1] * num_units)

    # Distribution amongst hampers should be random so we have multiple different
    # solutions
    rng = np.random.default_rng()
    rng.shuffle(item_arr)

    return item_arr


def make_random_chromosome(num_hampers: int, units: list[int]) -> np.ndarray:
    """Make a single randomised chromosome for the hamper problem GA.

    Args:
        num_hampers (int): Number of hampers that will be created.
        units (list[int]): List containing number of each item that is available

    returns:
        np.ndarray: Chromome made up of randomised binary arrays for each item
    """
    chromosome = [randomly_distribute_item(num_hampers, num_units) for num_units in units]

    return np.array(chromosome)


def initialise_population(num_hampers: int, item_amounts: list, pop_size: int) -> list:
    return [make_random_chromosome(num_hampers, item_amounts) for _ in range(pop_size)]

