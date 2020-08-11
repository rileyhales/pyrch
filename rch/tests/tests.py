import numpy as np
import pandas as pd

import rch


def test_idw():
    df = pd.DataFrame({'x': list(range(-20, 21)),
                       'y': np.random.randint(-20, 20, size=(41,)),
                       'v': np.random.randint(-60, 60, size=(41,))})
    print(df.loc[(df['x'] > 0) & (df['y'] > 0)])
    exit()
    print(rch.interpolate_idw(df.values, (0, 0), bound=1))
    return


test_idw()
