import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

EMPTY = 0
OBSTACLE = 1
TARGET = 2
START = 3


def extract_greedy_path(env, mc, max_steps=1000):
    state = env.reset()
    path = [(state[0], state[1])]

    for _ in range(max_steps):
        q_values = [mc.Q[(state, a)] for a in env.actions]
        best_action = env.actions[int(np.argmax(q_values))]

        next_state, _, done = env.step(best_action)
        state = next_state
        path.append((state[0], state[1]))

        if done:
            break

    return path


def plot_grid_with_path(grid, path, title, save_path):
    grid = np.array(grid)

    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap="gray_r")

    xs = [p[0] for p in path]
    ys = [p[1] for p in path]

    plt.plot(xs, ys, color="red", linewidth=2, label="Learned path")
    plt.scatter(xs[0], ys[0], c="green", s=80, label="Start")
    plt.scatter(xs[-1], ys[-1], c="blue", s=80, label="Target")

    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.gca().invert_yaxis()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_experiment_table(results, save_path):
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    return df
