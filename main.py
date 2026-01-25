import time
from grids import grid_simple, grid_medium, grid_hard
from environment import GridEnvironment
from mc_control import MonteCarloControl


def run_experiment():
    print("=== Monte Carlo Path Finding Experiment ===")

    # choose grid
    grid = grid_hard()
    env = GridEnvironment(grid)

    # create MC controller
    mc = MonteCarloControl(actions=env.actions)

    # training parameters
    num_episodes = 5000
    epsilon = 0.3
    epsilon_decay = 0.995

    start_time = time.time()

    episode_lengths = mc.train(
        env,
        num_episodes=num_episodes,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay
    )

    end_time = time.time()

    print("\n=== Training completed ===")
    print(f"Episodes: {num_episodes}")
    print(f"Training time: {end_time - start_time:.2f} seconds")
    print(f"Final Q-table size: {len(mc.Q)}")
    print(f"Average episode length (last 100): "
          f"{sum(episode_lengths[-100:]) / 100:.1f}")

    return episode_lengths, mc


if __name__ == "__main__":
    run_experiment()
