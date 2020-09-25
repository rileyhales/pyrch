import numpy as np
import pandas as pd

a = pd.read_csv('/sudoku.csv', names=list(range(1, 10)))
a.index = a.columns
print(a)
locations = np.where(a == 3, True, False)
options = np.array([True] * 81).reshape(locations.shape)
print(options)
print(locations)

for index, row in enumerate(np.rollaxis(locations, 0)):
    if any(row):
        options[index, :] = False
for index, column in enumerate(np.rollaxis(locations, 1)):
    if any(column):
        options[:, index] = False

print(options)
print(locations)


class Sudoku:
    def __init__(self, grid: str or np.array, **kwargs):
        a = pd.read_csv(grid, names=list(range(1, 10)))
        a.index = a.columns
        self.grid = a

        # todo get the dimensions and store them
        self.width = 9
        self.height = 9
        # create an annotations array shape (rows, columns, rows * columns)

        # toggle optional placement rules
        self.king_rule = kwargs.get('king_rule', False)
        self.knight_rule = kwargs.get('knight_rule', False)
        self.diagonal_rule = kwargs.get('diagonal_rule', False)

    def solve(self, number: int):
        def eliminate_rows(number):
            return
