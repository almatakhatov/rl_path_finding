import random
import numpy as np

# Cell types
EMPTY = 0
OBSTACLE = 1
TARGET = 2
START = 3


class GridEnvironment:
    def __init__(self, grid):
        self.grid = grid
        self.height, self.width = grid.shape

        self.x = None
        self.y = None
        self.vx = None
        self.vy = None

        self.done = False
        self.steps = 0

        self.start_positions = self._find_start_positions()

        self.actions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),  (0, 0),  (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        self.max_velocity = 2

    def _find_start_positions(self):
        positions = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] == START:
                    positions.append((x, y))

        if not positions:
            raise ValueError("No starting line defined in the grid.")

        return positions

    def reset(self):
        self.x, self.y = random.choice(self.start_positions)
        self.vx = 0
        self.vy = 0
        self.done = False
        self.steps = 0

        return self.get_state()

    def get_state(self):
        return (self.x, self.y, self.vx, self.vy)
    
    def _clip_velocity(self, v):
        return max(-self.max_velocity, min(self.max_velocity, v))
    
    def _on_start_line(self, x, y):
        return self.grid[y, x] == START

    def _is_zero_velocity_invalid(self, vx, vy, x, y):
        return vx == 0 and vy == 0 and not self._on_start_line(x, y)

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode has terminated. Call reset().")

        self.steps += 1

        dvx, dvy = action

        # velocity update
        new_vx = self._clip_velocity(self.vx + dvx)
        new_vy = self._clip_velocity(self.vy + dvy)

        # zero-velocity rule
        if self._is_zero_velocity_invalid(new_vx, new_vy, self.x, self.y):
            new_vx, new_vy = self.vx, self.vy

        # tentative new position
        new_x = self.x + new_vx
        new_y = self.y + new_vy

        # checking collision
        if self._check_path_collision(self.x, self.y, new_x, new_y):
            self._handle_collision()
            reward = -1
            return self.get_state(), reward, self.done

        # applying movement
        self.x = new_x
        self.y = new_y
        self.vx = new_vx
        self.vy = new_vy

        # termination check
        if self.grid[self.y, self.x] == TARGET:
            self.done = True
            reward = -1
            return self.get_state(), reward, self.done

        reward = -1
        return self.get_state(), reward, self.done

    def _check_collision(self, x, y):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True

        if self.grid[y, x] == OBSTACLE:
            return True

        return False

    def _check_path_collision(self, x0, y0, x1, y1):
        steps = max(abs(x1 - x0), abs(y1 - y0))

        for i in range(1, steps + 1):
            xi = x0 + int(round(i * (x1 - x0) / steps))
            yi = y0 + int(round(i * (y1 - y0) / steps))

            if self._check_collision(xi, yi):
                return True

        return False

    def _handle_collision(self):
        self.x, self.y = random.choice(self.start_positions)
        self.vx = 0
        self.vy = 0
