import numpy as np
import pandas as pd

EXAMPLE = np.array([[np.nan, 8, 4, 9, np.nan, np.nan, 7, 5, np.nan],
                    [3, np.nan, 6, 4, np.nan, 5, 2, np.nan, 8.],
                    [np.nan, 5, np.nan, np.nan, np.nan, 2, 4, np.nan, 6.],
                    [np.nan, 1, 5, np.nan, np.nan, 8, 9, np.nan, 2.],
                    [9, np.nan, 8, 6, np.nan, np.nan, np.nan, np.nan, 4.],
                    [7, 6, 3, np.nan, 4, 9, 1, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1.],
                    [np.nan, np.nan, np.nan, 5, 2, 7, 6, 4, 9.],
                    [np.nan, np.nan, np.nan, 1, np.nan, np.nan, np.nan, np.nan, np.nan]], )


class Sudoku:
    def __init__(self, grid: str or np.array, **kwargs):
        if isinstance(grid, str):
            self.grid = pd.read_csv(grid, names=list(range(1, 10))) \
                .replace(kwargs.get('blank', 0), np.nan) \
                .values
        elif isinstance(grid, np.array):
            self.grid = grid
        else:
            raise TypeError('The grid argument should be a str path to a csv file or a np 9x9 array')

        self.solution = self.grid
        self.is_solved = False

        # solver configs
        self.solution_path = ''
        self.max_iter = kwargs.get('max_iter', 50)

    def solve(self):
        assigned = True
        i = 0
        while assigned:
            print(i)
            made_progress = []
            for n in range(1, 10):
                options = self._annotate(n)
                made_progress.append(self._assign_rows(options, n))
                made_progress.append(self._assign_rows(options, n))
                made_progress.append(self._assign_boxes(options, n))
            assigned = any(made_progress)
            if np.nansum(self.solution) == 9 * 45:
                print('solved')
                self.is_solved = True
                return self.solution
            elif i > self.max_iter:
                print('exceeded 50 iterations')
                return
            i += 1

        print('out of logical solution steps')
        print(self.solution_path)
        return self.solution

    def _annotate(self, n: int):
        # all empty squares are options, if they have a number in them then they are not
        options = np.where(np.isnan(self.solution), True, False)

        number_mask = np.where(self.solution == n, True, False)
        # annotate rows based on known locations
        for idx, row in enumerate(np.rollaxis(number_mask, 0)):
            if any(row):
                options[idx, :] = False
        # annotate columns based on known locations
        for idx, col in enumerate(np.rollaxis(number_mask, 1)):
            if any(col):
                options[:, idx] = False
        # annotate boxes based on known locations (if the num is in the box, other box cells aren't options)
        for _3row_idx in range(3):
            for _3col_idx in range(3):
                row_slice = slice(3 * _3row_idx, 3 * (_3row_idx + 1))
                col_slice = slice(3 * _3col_idx, 3 * (_3col_idx + 1))
                if np.any(self.solution[row_slice, col_slice] == n):
                    options[row_slice, col_slice] = False

        return options

    def _assign_rows(self, options: np.array, n: int):
        # assign rows: if there is only 1 true in the row, it must go there
        made_assignment = False
        for idx, row in enumerate(np.rollaxis(options, 0)):
            unq_vals, unq_idx, unq_count = np.unique(row, return_index=True, return_counts=True)
            if True in unq_vals and unq_count[-1] == 1:
                self.solution[idx, unq_idx[-1]] = n
                self.solution_path += f'assigned {n} at grid cell ({idx}, {unq_idx[-1]}) -> row unique\n'
                made_assignment = True
        return made_assignment

    def _assign_cols(self, options: np.array, n: int):
        # assign cols: if there is only 1 true in the column, it must go there
        made_assignment = False
        for idx, col in enumerate(np.rollaxis(options, 1)):
            unq_vals, unq_idx, unq_count = np.unique(col, return_index=True, return_counts=True)
            if True in unq_vals and unq_count[-1] == 1:
                self.solution[unq_idx[-1], idx] = n
                self.solution_path += f'assigned {n} at grid cell ({unq_idx[-1]}, {idx}) -> col unique\n'
                made_assignment = True
        return made_assignment

    def _assign_boxes(self, options: np.array, n: int):
        # assign boxes: if there is only 1 true in the box, it must go there
        made_assignment = False
        for _3row_idx in range(3):
            for _3col_idx in range(3):
                row_slice = slice(3 * _3row_idx, 3 * (_3row_idx + 1))
                col_slice = slice(3 * _3col_idx, 3 * (_3col_idx + 1))
                unq_vals, unq_idx, unq_count = np.unique(
                    options[row_slice, col_slice], return_index=True, return_counts=True)
                if True in unq_vals and unq_count[-1] == 1:
                    if unq_idx[-1] >= 6:
                        row = 3 * _3row_idx + 2
                        col = 3 * _3col_idx + unq_idx[-1] - 6
                    elif unq_idx[-1] >= 3:
                        row = 3 * _3row_idx + 1
                        col = 3 * _3col_idx + unq_idx[-1] - 3
                    else:
                        row = 3 * _3row_idx
                        col = 3 * _3col_idx + unq_idx[-1]
                    self.solution_path += f'assigned {n} at grid cell ({row}, {col}) -> box unique\n'
                    self.solution[row, col] = n
                    made_assignment = True
        return made_assignment

    def solution_to_csv(self, path: str):
        pd.DataFrame(self.solution).to_csv(path, index=False, header=False)
