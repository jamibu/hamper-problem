import random

import numpy as np

import mutation


def test_mutate():
    # Hamper=2, zero_idx=0, one_idx=0
    # i.e. expect the first 0 to be switched with the first 1 in third hamper
    random.seed(42)
    chromosome = np.array([[1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 0]])
    expected = np.array([[1, 0, 1, 0], [0, 0, 1, 1], [0, 1, 1, 0]])

    result = mutation.mutate(chromosome)
    np.testing.assert_array_equal(result, expected)

