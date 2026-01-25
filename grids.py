import numpy as np

EMPTY = 0
OBSTACLE = 1
TARGET = 2
START = 3


def grid_simple():
    """
    Simple difficulty grid with a few obstacles
    """
    grid = np.zeros((10, 10), dtype=int)

    grid[4, 3:7] = OBSTACLE
    grid[6, 2:5] = OBSTACLE

    grid[0, 8] = TARGET

    grid[9, :] = START

    return grid


def grid_medium():
    """
    Medium difficulty grid 
    """
    grid = np.zeros((12, 12), dtype=int)

    grid[3, 2:10] = OBSTACLE
    grid[6, 1:8] = OBSTACLE
    grid[8, 4:11] = OBSTACLE

    grid[0, 10] = TARGET
    grid[11, :] = START

    return grid


def grid_hard():
    """
    Hard difficulty grid with a lot of obstacles
    """
    grid = np.zeros((15, 15), dtype=int)

    grid[2, 1:14] = OBSTACLE
    grid[5, 0:13] = OBSTACLE
    grid[8, 2:15] = OBSTACLE
    grid[11, 1:14] = OBSTACLE

    grid[0, 13] = TARGET
    grid[14, :] = START

    return grid