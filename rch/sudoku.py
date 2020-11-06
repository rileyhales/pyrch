import numpy as np
import pandas as pd


class Sudoku:
    def __init__(self, grid: str or np.ndarray, **kwargs):
        if isinstance(grid, str):
            self.grid = pd.read_csv(grid, names=list(range(1, 10))).replace(kwargs.get('blank', 0), np.nan).values
        elif isinstance(grid, np.ndarray):
            self.grid = grid
        else:
            raise TypeError('The grid argument should be a str path to a csv file or a np 9x9 array')

        self.solution = self.grid
        self.is_solved = False

        # solver configs
        self.steps = ''
        self.iter = 0
        self.max_iter = kwargs.get('max_iter', 50)

    def solve(self):
        assigned = True
        self.iter = 0
        while assigned:
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
            elif self.iter > self.max_iter:
                print('exceeded 50 iterations')
                return
            self.iter += 1

        print('out of logical solution steps')
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

                # if the number is in the box, none of the other cells need to be considered options
                if np.any(self.solution[row_slice, col_slice] == n):
                    options[row_slice, col_slice] = False

                # extrapolate remaining possibilities
                box = options[row_slice, col_slice].reshape(3, 3)
                num_true = sum(box.flatten())

                # the following extrapolations only apply if there are 2 or 3 options in the box
                if num_true not in (2, 3):
                    continue

                # if all the options are in the same row/col of the box then number will go in that row/col of the box
                # so we can rule out other open spaces in the row of the sudoku grid
                for idx, row in enumerate(np.rollaxis(box, 0)):
                    if sum(row) == num_true:
                        columns_to_rule_out = list(range(9))
                        columns_to_rule_out.remove(3 * _3col_idx)
                        columns_to_rule_out.remove(3 * _3col_idx + 1)
                        columns_to_rule_out.remove(3 * _3col_idx + 2)
                        for c in columns_to_rule_out:
                            options[3 * _3row_idx + idx, c] = False
                for idx, col in enumerate(np.rollaxis(box, 1)):
                    if sum(col) == num_true:
                        rows_to_rule_out = list(range(9))
                        rows_to_rule_out.remove(3 * _3row_idx)
                        rows_to_rule_out.remove(3 * _3row_idx + 1)
                        rows_to_rule_out.remove(3 * _3row_idx + 2)
                        for r in rows_to_rule_out:
                            options[r, 3 * _3col_idx + idx] = False

        return options

    def _assign_rows(self, options: np.array, n: int):
        # assign rows: if there is only 1 true in the row, it must go there
        made_assignment = False
        for idx, row in enumerate(np.rollaxis(options, 0)):
            unq_vals, unq_idx, unq_count = np.unique(row, return_index=True, return_counts=True)
            if True in unq_vals and unq_count[-1] == 1:
                self.solution[idx, unq_idx[-1]] = n
                self.steps += f'assigned {n} at grid cell ({idx}, {unq_idx[-1]}) -> row unique\n'
                made_assignment = True
        return made_assignment

    def _assign_cols(self, options: np.array, n: int):
        # assign cols: if there is only 1 true in the column, it must go there
        made_assignment = False
        for idx, col in enumerate(np.rollaxis(options, 1)):
            unq_vals, unq_idx, unq_count = np.unique(col, return_index=True, return_counts=True)
            if True in unq_vals and unq_count[-1] == 1:
                self.solution[unq_idx[-1], idx] = n
                self.steps += f'assigned {n} at grid cell ({unq_idx[-1]}, {idx}) -> col unique\n'
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
                    self.steps += f'assigned {n} at grid cell ({row}, {col}) -> box unique\n'
                    self.solution[row, col] = n
                    made_assignment = True
        return made_assignment

    def to_csv(self, path: str) -> None:
        pd.DataFrame(self.solution).astype(int).to_csv(path, index=False, header=False)

    def to_html(self) -> str:
        return pd.DataFrame(self.solution).astype(int).to_html(index=False, header=False)
