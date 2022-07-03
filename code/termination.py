from numpy.typing import NDArray


def terminate(
    fitness: NDArray,
    target: int,
) -> bool:
    if fitness <= target:
        result = True
    else:
        result = False

    return result

