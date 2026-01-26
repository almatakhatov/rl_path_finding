import numpy as np
import matplotlib.pyplot as plt

EMPTY = 0
OBSTACLE = 1
TARGET = 2
START = 3

def extract_greedy_path(env, mc, max_steps=500):
    """
    Follow the greedy policy (epsilon = 0) and record the path.
    Returns list of (x, y) positions.
    """
    state = env.reset()
    path = [(state[0], state[1])]

    for _ in range(max_steps):
        # choose best action
        q_values = []
        for action in env.actions:
            q_values.append(mc.Q[(state, action)])

        max_q = max(q_values)
        best_actions = [
            action for action, q in zip(env.actions, q_values)
            if q == max_q
        ]

        action = best_actions[0]  # deterministic greedy
        next_state, reward, done = env.step(action)

        state = next_state
        path.append((state[0], state[1]))

        if done:
            break

    return path

def plot_grid_with_path(grid, path, title="Learned Path", save_path=None):
    grid = np.array(grid)

    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap="gray_r")

    xs = [p[0] for p in path]
    ys = [p[1] for p in path]

    plt.plot(xs, ys, color="red", linewidth=2, label="Path")
    plt.scatter(xs[0], ys[0], color="green", s=100, label="Start")
    plt.scatter(xs[-1], ys[-1], color="blue", s=100, label="Target")

    plt.title(title)
    plt.legend()
    plt.gca().invert_yaxis()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()

def plot_training_curve(episode_lengths, title="Training Curve", save_path=None):
    plt.figure(figsize=(8, 4))
    plt.plot(episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title(title)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()

