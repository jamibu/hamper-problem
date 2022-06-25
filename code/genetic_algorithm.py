import pandas as pd
import numpy as np
from numpy.typing import NDArray

from initialise import initialise_population
from selection import fitness_calc, selection
from crossover import crossover
from mutation import mutate


def main():
    # GA Settings
    pop_size = 50
    num_generations = 100

    # Problem inputs
    num_hampers = 25
    target = 5000
    item_data = pd.read_csv("../CharityBulkPurchaseList")

    # Top half of solutions will be used to create  new solutions
    num_parents = int(pop_size / 2)

    # Initialise population
    population = initialise_population(
        num_hampers,
        item_data["price per unit"].tolist(),
        pop_size,
    )

    # Run optimisation loop
    units = item_data["total units"].values
    price = item_data["price per unit"].values
    for i in range(1, num_generations + 1):
        # Determine fitness of current generation
        fitness = [fitness_calc(c, price, target) for c in population]

        # Check if termination critera are met

        # Select fittest solutions to create new solutions
        parents = selection(fitness, num_parents, population)

    # Display results

