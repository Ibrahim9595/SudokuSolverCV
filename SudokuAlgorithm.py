from math import sqrt

GRID_SIZE = 9
CELLS_N = int(sqrt(GRID_SIZE))


class SudokuAlgorithm:
    def __init__(self, grid):
        self.__grid = grid
        self.__state = self.__initializeMainState()
        self.__soluion = None

    def __initializeMainState(self):
        rows = [dict() for i in range(GRID_SIZE)]
        cols = [dict() for i in range(GRID_SIZE)]
        cells = [dict() for i in range(GRID_SIZE)]

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                ptr = self.__grid[i][j]
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

    def __getNextState(self, board, state, row, col, val):
        next_grid = [[board[j][i]
                      for i in range(GRID_SIZE)] for j in range(GRID_SIZE)]
        next_grid[row][col] = val

        next_state = {
            "rows": [row.copy() for row in state['rows']],
            "cols": [col.copy() for col in state['cols']],
            "cells": [cell.copy() for cell in state['cells']]
        }

        c_idx = (row // CELLS_N) * CELLS_N + (col // CELLS_N)

        next_state['rows'][row][val] = True
        next_state['cols'][col][val] = True
        next_state['cells'][c_idx][val] = True

        return (next_grid, next_state)

    def solve(self):
        self.__backTrack(0, 0, self.__grid, self.__state)
        return self.__soluion

    def __backTrack(self, row, col, board, state):
        if row >= GRID_SIZE:
            self.__soluion = board
            return True

        next_row = row
        if col + 1 >= GRID_SIZE:
            next_row = row + 1

        next_col = (col + 1) % GRID_SIZE

        if board[row][col] != 0:
            return self.__backTrack(next_row, next_col, board, state)

        for i in range(1, GRID_SIZE + 1):
            c_idx = (row // CELLS_N) * CELLS_N + (col // CELLS_N)

            if(
                (i not in state['rows'][row]) and
                (i not in state['cols'][col]) and
                (i not in state['cells'][c_idx])
            ):
                (next_board, next_state) = self.__getNextState(
                    board, state, row, col, i)
                ret = self.__backTrack(
                    next_row, next_col, next_board, next_state)
                if ret:
                    return True

        return False
