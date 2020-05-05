from math import sqrt

GRID_SIZE = 9
CELLS_N = int(sqrt(GRID_SIZE))


class SudokuAlgorithm:
    def __init__(self, grid):
        self.grid = grid
        self.state = self.__initializeMainState()

    def __initializeMainState(self):
        rows = [dict() for i in range(GRID_SIZE)]
        cols = [dict() for i in range(GRID_SIZE)]
        cells = [dict() for i in range(GRID_SIZE)]

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                ptr = self.grid[i][j]
                if ptr:
                    # cell index
                    c_idx = (i // CELLS_N) * CELLS_N + (j // CELLS_N)
                    self.__validState(rows[i], cols[j], cells[c_idx], ptr)
                    rows[i][ptr] = True
                    cols[j][ptr] = True
                    cells[c_idx][ptr] = True
        return {
            "rows": rows,
            "cols": cols,
            "cells": cells
        }

    def __validState(self, row, col, cell, number):
        if (number in row) or (number in col) or (number in cell):
            raise(Exception('Invalid sudoku grid'))

    def solve(self):
        pass
