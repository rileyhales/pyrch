import unittest

import pandas as pd

import rch


class TestMisc(unittest.TestCase):
    print('hello')

    def gen_grid(self):
        print('gen grid')
        self.assertIsInstance(rch.misc.gen_interpolation_grid(True, 4), pd.DataFrame, "Failed random grid")
        self.assertIsInstance(rch.misc.gen_interpolation_grid(False, 4), pd.DataFrame, "Failed structured grid")

    def idw(self):
        print('idw')
        grid = rch.misc.gen_interpolation_grid()


if __name__ == '__main__':
    unittest.main()
