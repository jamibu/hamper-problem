import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy.typing import NDArray


def plot_value_vs_hampers(
    hampers: NDArray,
    total_item_value: int,
    target_hamper_value: int
):
    # Calculate best possible value of hamper
    cost_diff = np.abs(total_item_value - hampers * target_hamper_value)

    # Make the plot
    _, ax = plt.subplots(figsize=[8, 4])
    sns.scatterplot(x=hampers, y=cost_diff, ax=ax)
    # Show each hamper
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("Number of hampers")
    ax.set_ylabel("Sum of abs diff from 5000")
    plt.show()


def plot_fitness():
    pass
