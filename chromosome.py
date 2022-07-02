import pandas as pd
import numpy as np
import


class Chromosome:

    def __init__(self, chromosome: np.ndarray, item_data):

        pass

    @classmethod
    def create_random(cls, num_hampers: int, item_data: pd.DataFrame):
        """Randomly create a valid chromosome.

        Args:
            num_hampers (int): Number of hampers that items will be assigned to.
            item_data (pd.DataFrame): Pandas dataframe containing
        """
        chromosome = [randomly_distribute_item(num_hampers, num_units) for num_units in units]
        return Chromosome(np.array(chromosome), item_df)

    def calc_fitness(self):
        pass

    def mutate(self):
        pass

    def display(self):
        pass



