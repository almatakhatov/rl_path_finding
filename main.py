import os
import time

from environment import GridEnvironment
from mc_control import MonteCarloControl
from grids import grid_layout_1, grid_layout_2, grid_layout_3, grid_layout_4
from visualization import extract_greedy_path, plot_grid_with_path, save_experiment_table 


def run_experiment(grid_fn, grid_name, output_dir="plots"):
    print(f"\nRunning training on {grid_name}")

    os.makedirs(output_dir, exist_ok=True)

    env = GridEnvironment(grid_fn())
    mc = MonteCarloControl(actions=env.actions)

    start_time = time.time()

    episode_lengths = mc.train(
        env,
        num_episodes=5000,
        epsilon=0.3,
        epsilon_decay=0.995
    )

    runtime = time.time() - start_time
    avg_steps = sum(episode_lengths[-100:]) / 100

    path = extract_greedy_path(env, mc)

    plot_grid_with_path(
        grid_fn(),
        path,
        title=f"Learned Path – {grid_name}",
        save_path=f"{output_dir}/{grid_name}_path.png"
    )

    print(f"Runtime: {runtime:.2f}s | Avg steps: {avg_steps:.1f} | Q size: {len(mc.Q)}")

    return {
        "grid": grid_name,
        "runtime": runtime,
        "avg_steps": avg_steps,
        "q_size": len(mc.Q)
    }


def main():
    grids = [
        ("Grid-1", grid_layout_1),
        ("Grid-2", grid_layout_2),
        ("Grid-3", grid_layout_3),
        ("Grid-4", grid_layout_4),
    ]

    results = []

    for name, grid_fn in grids:
        results.append(run_experiment(grid_fn, name))

    save_experiment_table(
        results,
        save_path="plots/experiment_summary.csv"
    )


if __name__ == "__main__":
    main()
