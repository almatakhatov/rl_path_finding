import os
import time
from environment import GridEnvironment
from mc_control import MonteCarloControl
from grids import grid_layout_1, grid_layout_2, grid_layout_3, grid_layout_4, grid_layout_5
from visualization import extract_greedy_path, plot_grid_with_path, plot_training_curve

def run_full_experiment(grid_fn, grid_name, output_dir="plots"):
    print(f"\n=== Running experiment on {grid_name} ===")

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
    avg_last_100 = sum(episode_lengths[-100:]) / 100

    print(f"Training time: {runtime:.2f}s")
    print(f"Average episode length (last 100): {avg_last_100:.1f}")
    print(f"Q-table size: {len(mc.Q)}")

    # ---- Save training curve ----
    plot_training_curve(
        episode_lengths,
        title=f"Training Curve – {grid_name}",
        save_path=f"{output_dir}/{grid_name}_training_curve.png"
    )

    # ---- Save learned path ----
    path = extract_greedy_path(env, mc)
    plot_grid_with_path(
        grid_fn(),
        path,
        title=f"Learned Path – {grid_name}",
        save_path=f"{output_dir}/{grid_name}_learned_path.png"
    )

    return {
        "grid": grid_name,
        "runtime": runtime,
        "avg_steps": avg_last_100,
        "q_size": len(mc.Q)
    }


def main():
    grids = [
        ("Grid-1", grid_layout_1),
        ("Grid-2", grid_layout_2),
        ("Grid-3", grid_layout_3),
        ("Grid-4", grid_layout_4),
        ("Grid-5", grid_layout_5),
    ]

    results = []

    for name, grid_fn in grids:
        result = run_full_experiment(grid_fn, name)
        results.append(result)

    print("\n=== Final Experiment Summary ===")
    for r in results:
        print(
            f"{r['grid']:8s} | "
            f"Runtime: {r['runtime']:.2f}s | "
            f"Avg steps: {r['avg_steps']:.1f} | "
            f"Q size: {r['q_size']}"
        )


if __name__ == "__main__":
    main()
