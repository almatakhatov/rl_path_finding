import numpy as np

EMPTY = 0
OBSTACLE = 1
TARGET = 2
START = 3


def grid_layout_1():
    grid = np.zeros((10, 10), dtype=int)

    grid[4, 6:10] = OBSTACLE
    grid[6, 2:5] = OBSTACLE

    grid[0, 8] = TARGET

    grid[9, :] = START

    return grid

def grid_layout_2():
    grid = np.zeros((10, 10), dtype=int)

    grid[6, 8:10] = OBSTACLE
    grid[3, 0:5] = OBSTACLE
    grid[4:8, 6] = OBSTACLE
    
    grid[9, :] = START

    grid[0, 9] = TARGET
  
    return grid


def grid_layout_3():
    grid = np.zeros((12, 12), dtype=int)

    grid[3, 6:13] = OBSTACLE
    grid[6, 1:8] = OBSTACLE
    grid[8, 4:11] = OBSTACLE

    grid[0, 10] = TARGET
    grid[11, :] = START

    return grid

def grid_layout_4():
    grid = np.zeros((12, 12), dtype=int)
    
    grid[8, 5:13] = OBSTACLE
    grid[5, 0:9] = OBSTACLE
    
    
    grid[11, :] = START

    grid[0, 11] = TARGET

    return grid
