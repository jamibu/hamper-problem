from random import randint, uniform
import numpy as np


def make_offspring(
    parent1: np.ndarray,
    parent2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # Flat arrays make it easier to split and combine the parent arrays
    x, y = parent1.shape
    parent1_flat = parent1.reshape(x * y)
    parent2_flat = parent2.reshape(x * y)

    # Point at which we crossover should be random (can't be at the very ends though)
    cross = randint(1, parent1_flat.shape[0] - 1)
    print(cross)

    # Crossover to make offspring
    offspring1_flat = np.append(parent1_flat[0:cross], parent2_flat[cross:])
    offspring2_flat = np.append(parent2_flat[0:cross], parent1_flat[cross:])

    # Return offspring with original 2D shape
    return (offspring1_flat.reshape(x, y), offspring2_flat.reshape(x, y))

