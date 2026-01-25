import random
from grids import grid_simple
from environment import GridEnvironment


def test_environment_full():
    print("=== Phase 1 Environment Test ===")

    env = GridEnvironment(grid_simple())

    state = env.reset()
    print(f"Initial state: {state}")
    assert env.grid[state[1], state[0]] == 3, "Agent did not start on start line"
    assert state[2] == 0 and state[3] == 0, "Initial velocity not zero"

    for step in range(500):
        action = random.choice(env.actions)
        next_state, reward, done = env.step(action)

        x, y, vx, vy = next_state

        print(
            f"Step {step:03d} | "
            f"Action={action} | "
            f"State={next_state} | "
            f"Reward={reward} | "
            f"Done={done}"
        )

        # reward must always be -1
        assert reward == -1, "Reward is not -1"

        # velocity bounds check
        assert abs(vx) <= env.max_velocity, "vx exceeds max velocity"
        assert abs(vy) <= env.max_velocity, "vy exceeds max velocity"

        # zero velocity allowed only on start line
        if vx == 0 and vy == 0:
            assert env.grid[y, x] == 3, "(0,0) velocity off start line"

        # position must always be inside grid
        assert 0 <= x < env.width, "x out of bounds"
        assert 0 <= y < env.height, "y out of bounds"

        if done:
            print("\nTarget reached. Episode terminated correctly.")
            break

    else:
        print("\nTarget not reached within 500 random steps (this is OK).")

    print("=== Phase 1 Test Completed ===")


if __name__ == "__main__":
    test_environment_full()
