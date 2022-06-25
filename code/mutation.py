from random import randint, uniform

import numpy as np
from numpy.typing import NDArray


def mutate(chromosome: NDArray):
    # Want to mutate a value in a single hamper
    hamper_idx = randint(0, chromosome.shape[0] - 1)
    hamper = chromosome[hamper_idx, :]

    # Will switch a 0 with a 1 within a hamper
    zeroes = np.where(hamper==0)[0]
    ones = np.where(hamper==1)[0]

    # Select which 0 and which 1 to switch
    zero_to_switch = randint(0, zeroes.shape[0] - 1)
    one_to_switch = randint(0, ones.shape[0] - 1)

    zero_idx = zeroes[zero_to_switch]
    one_idx = ones[one_to_switch]

    # Swap a randomly selected 1 and 0 within the hamper
    hamper[zero_idx] = 1
    hamper[one_idx] = 0

    return chromosome

