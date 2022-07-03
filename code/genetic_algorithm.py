from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.typing import NDArray

from initialise import initialise_population
from selection import fitness_calc, selection
from termination import terminate
from twopoint_crossover import crossover
from mutation import mutate


def main():
    # GA Settings
    pop_size = 250
    num_generations = 1000

    # Terminate after this many generations with no improvement
    target_fitness = 1709

    # Problem inputs
    num_hampers = 25
    target = 5000
    item_data = pd.read_csv("../CharityBulkPurchaseList.csv")

    # Top half of solutions will be used to create  new solutions
    num_parents = int(pop_size / 2)
    num_offspring = pop_size - num_parents

    # Initialise population
    population = initialise_population(
        num_hampers,
        item_data["total units"].tolist(),          # type: ignore
        pop_size,
    )

    # Run optimisation loop
    units = item_data["total units"].values         # type: ignore
    price = item_data["price per unit"].values      # type: ignore

    mean_fitness = []
    best_fitness = []

    for _ in tqdm(range(1, num_generations + 1)):
        # Determine fitness of current generation
        fitness = np.array([fitness_calc(c, price, target) for c in population])

        best_solution = fitness.min()

        mean_fitness.append(fitness.mean())
        best_fitness.append(best_solution)

        # Check if termination critera are met
        if terminate(best_solution, target_fitness):
            break

        # Select fittest solutions to create new solutions
        parents = selection(fitness, num_parents, population)

        # Do crossover to produce new solutions
        offspring = crossover(parents, num_offspring, units)

        for chromosome in offspring:
            np.testing.assert_array_equal(chromosome.sum(axis=1), units)

        # Mutate some offspring
        mutants = mutate(offspring, mutation_rate=0.5)

        # Create new population with fittest parents and new solutions
        population = parents + mutants

    # Determine fitness of current generation
    fitness = np.array([fitness_calc(c, price, target) for c in population])

    best_index = fitness.argmin()
    best_solution = population[best_index]

    print()
    print("Best Fitness")
    print(fitness[best_index])

    for i in range(best_solution.shape[1]):
        display_hamper(
            i,
            best_solution[:, i],
            item_data["item"].values,
            item_data["price per unit"].values
        )

    _, ax = plt.subplots()
    ax.plot(mean_fitness, label="Mean Fitness")
    ax.plot(best_fitness, label="Best Solution")

    plt.show()


def display_hamper(
    hamper_num: int,
    hamper: NDArray,
    item_names: NDArray,
    item_values: NDArray
):
    hamper_items = item_names[hamper == 1].tolist()
    hamper_value = (item_values * hamper).sum()

    print(f"{hamper_num}: {int(hamper_value) : >2} - {hamper_items}")


if __name__ == "__main__":
    main()

