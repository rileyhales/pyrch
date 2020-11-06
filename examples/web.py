import numpy as np
from googleapiclient.discovery import build

import rch

service = build('sheets', 'v4', developerKey='AIzaSyC2dkypmio4HAXpS1jL2xU-2kVqva4hdyY')
sheet_id = '1Iw7J_aOsbjcPvRq97eNWNiSbn_GUlywZ9cNR3du1miE'
sheet_range = 'sudoku_puzzle!A:J'
a = rch.web.read_google_sheet(service, sheet_id, sheet_range, columns=True, indexed=True)
a = a.replace('', np.nan)
sudoku_puzzle = rch.sudoku.Sudoku(a.values.astype(np.float))
sudoku_puzzle.solve()
print(sudoku_puzzle.to_html())
