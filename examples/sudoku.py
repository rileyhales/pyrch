import numpy as np

import rch

EXAMPLE_EASY = np.array([[np.nan, 8, 4, 9, np.nan, np.nan, 7, 5, np.nan],
                         [3, np.nan, 6, 4, np.nan, 5, 2, np.nan, 8],
                         [np.nan, 5, np.nan, np.nan, np.nan, 2, 4, np.nan, 6],
                         [np.nan, 1, 5, np.nan, np.nan, 8, 9, np.nan, 2],
                         [9, np.nan, 8, 6, np.nan, np.nan, np.nan, np.nan, 4],
                         [7, 6, 3, np.nan, 4, 9, 1, np.nan, np.nan],
                         [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1],
                         [np.nan, np.nan, np.nan, 5, 2, 7, 6, 4, 9],
                         [np.nan, np.nan, np.nan, 1, np.nan, np.nan, np.nan, np.nan, np.nan]], )

EXAMPLE_HARD = np.array([[np.nan, 6, np.nan, np.nan, np.nan, 7, np.nan, 1, 8],
                         [np.nan, np.nan, 2, np.nan, np.nan, np.nan, np.nan, np.nan, 5],
                         [np.nan, np.nan, np.nan, np.nan, 6, np.nan, np.nan, np.nan, 4],
                         [np.nan, np.nan, np.nan, 1, 9, np.nan, np.nan, 3, np.nan],
                         [np.nan, np.nan, 1, 4, np.nan, 5, 6, np.nan, np.nan],
                         [np.nan, 7, np.nan, np.nan, 2, 8, np.nan, np.nan, np.nan],
                         [6, np.nan, np.nan, np.nan, 1, np.nan, np.nan, np.nan, np.nan],
                         [7, np.nan, np.nan, np.nan, np.nan, np.nan, 8, np.nan, np.nan],
                         [4, 1, np.nan, 7, np.nan, np.nan, np.nan, 2, np.nan]], )

puzzle = rch.sudoku.Sudoku(EXAMPLE_HARD)
print(puzzle.solve())
