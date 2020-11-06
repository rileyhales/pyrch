import unittest

import pandas as pd

import rch


class TestMisc(unittest.TestCase):

    def gen_grid(self):
        self.assertIsInstance(rch.eng.gen_interpolation_grid(True, 4), pd.DataFrame, "Failed random grid")
        self.assertIsInstance(rch.eng.gen_interpolation_grid(False, 4), pd.DataFrame, "Failed structured grid")

    def idw(self):
        grid = rch.eng.gen_interpolation_grid()


if __name__ == '__main__':
    unittest.main()
