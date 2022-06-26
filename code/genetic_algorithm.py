import pandas as pd
import numpy as np

from initialise import initialise_population
from selection import fitness_calc, selection
from crossover import crossover
from mutation import mutate


def main():
    # GA Settings
    pop_size = 100
    num_generations = 1000

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
    for i in range(1, num_generations + 1):
        print(f"Generation: {i}\r", end="")
        # Determine fitness of current generation
        fitness = np.array([fitness_calc(c, price, target) for c in population])

        # Check if termination critera are met

        # Select fittest solutions to create new solutions
        parents = selection(fitness, num_parents, population)

        # Do crossover to produce new solutions
        offspring = crossover(parents, num_offspring, units, price)

        print()
        for chromosome in offspring:
            np.testing.assert_array_equal(chromosome.sum(axis=1), units)

        # Mutate some offspring
        mutants = mutate(offspring, mutation_rate=0.7)

        # Create new population with fittest parents and new solutions
        population = parents + mutants

    # Determine fitness of current generation
    fitness = np.array([fitness_calc(c, price, target) for c in population])

    best_index = fitness.argmin()
    best_solution = population[best_index]

    print()
    print("Best Fitness")
    print(fitness[best_index])

    print(best_solution.sum(axis=1))
    print(item_data)


if __name__ == "__main__":
    main()

