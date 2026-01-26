import numpy as np

EMPTY = 0
OBSTACLE = 1
TARGET = 2
START = 3


def grid_layout_1():
    grid = np.zeros((10, 10), dtype=int)

    grid[4, 3:7] = OBSTACLE
    grid[6, 2:5] = OBSTACLE

    grid[0, 8] = TARGET

    grid[9, :] = START

    return grid

def grid_layout_2():
    grid = np.zeros((10, 10), dtype=int)

    grid[6, 2:8] = OBSTACLE
    grid[3, 1:6] = OBSTACLE
    grid[4:8, 5] = OBSTACLE
    
    grid[9, :] = START

    grid[0, 9] = TARGET
  
    return grid


def grid_layout_3():
    grid = np.zeros((12, 12), dtype=int)

    grid[3, 2:10] = OBSTACLE
    grid[6, 1:8] = OBSTACLE
    grid[8, 4:11] = OBSTACLE

    grid[0, 10] = TARGET
    grid[11, :] = START

    return grid

def grid_layout_4():
    grid = np.zeros((12, 12), dtype=int)
    
    grid[8, 2:10] = OBSTACLE
    grid[6, 1:9] = OBSTACLE
    grid[4, 3:11] = OBSTACLE
    grid[5:9, 4] = OBSTACLE
    grid[2:6, 7] = OBSTACLE
    
    grid[11, :] = START

    grid[0, 11] = TARGET

    return grid

def grid_layout_5():
    grid = np.zeros((14, 14), dtype=int)

    grid[10, 2:12] = OBSTACLE
    grid[7, 1:11] = OBSTACLE
    grid[4, 3:13] = OBSTACLE
    grid[5:11, 6] = OBSTACLE
    grid[2:8, 9] = OBSTACLE

    grid[13, :] = START

    grid[0, 13] = TARGET

    return grid
