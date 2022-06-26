import numpy as np


def fitness_calc(
    chromosome: np.ndarray,
    item_values: np.ndarray,
    target_hamper_value: float
) -> float:
    # Since our genes are binary multiplying by an array of values
    # replaces 1s with the value of the item
    hamper_value = np.dot(item_values, chromosome)

    # Aim is to minimise the sum of the absolute diff therefore smaller is better
    diff = np.abs(hamper_value - target_hamper_value)
    return diff.sum()


def selection(fitness: np.ndarray, num_parents: int, population: list) -> list:
    """Select the fittest solutions to use as parents for the next generation."""
    population_arr = np.array(population)
    fitness_idx = fitness.argsort()
    sorted_population = population_arr[fitness_idx]

    return list(sorted_population[0:num_parents])

