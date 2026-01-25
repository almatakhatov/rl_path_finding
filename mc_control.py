import random
from collections import defaultdict

class MonteCarloControl:
    def __init__(self, actions, gamma=1.0):
        """
        actions: list of all possible actions (vx, vy)
        gamma: discount factor (1.0 for episodic task)
        """
        self.actions = actions
        self.gamma = gamma

        # Q-table: maps (state, action) -> value
        self.Q = defaultdict(float)

        # visit counter: maps (state, action) -> count
        self.N = defaultdict(int)

    def epsilon_greedy_action(self, state, epsilon):
        """
        Select an action using epsilon-greedy policy.
        """
        # exploration
        if random.random() < epsilon:
            return random.choice(self.actions)

        # exploitation: choose action with max Q-value
        q_values = []
        for action in self.actions:
            q_values.append(self.Q[(state, action)])

        max_q = max(q_values)

        # handle ties randomly
        best_actions = [
            action for action, q in zip(self.actions, q_values)
            if q == max_q
        ]

        return random.choice(best_actions)

    def generate_episode(self, env, epsilon, max_steps=1000):
        """
        Generate one episode using epsilon-greedy policy.
        Returns a list of (state, action, reward).
        """
        episode = []

        state = env.reset()

        for _ in range(max_steps):
            action = self.epsilon_greedy_action(state, epsilon)
            next_state, reward, done = env.step(action)

            episode.append((state, action, reward))

            state = next_state

            if done:
                break

        return episode

    def compute_returns(self, episode):
        """
        Compute returns G_t for each (state, action) in the episode.
        Returns a list of (state, action, G).
        """
        returns = []
        G = 0

        # traverse episode backwards
        for state, action, reward in reversed(episode):
            G = reward + self.gamma * G
            returns.append((state, action, G))

        # reverse back to original order
        returns.reverse()
        return returns

    def update_Q(self, returns):
        """
        Update Q-values using every-visit Monte Carlo.
        """
        for state, action, G in returns:
            self.N[(state, action)] += 1
            n = self.N[(state, action)]

            # incremental average update
            self.Q[(state, action)] += (G - self.Q[(state, action)]) / n

    def train(
        self,
        env,
        num_episodes=5000,
        epsilon=0.2,
        epsilon_decay=None,
        min_epsilon=0.01,
        max_steps_per_episode=1000
    ):
        """
        Train using Monte Carlo control.

        Returns a list of episode lengths.
        """
        episode_lengths = []

        for episode_idx in range(num_episodes):
            # generate episode
            episode = self.generate_episode(
                env,
                epsilon=epsilon,
                max_steps=max_steps_per_episode
            )

            episode_lengths.append(len(episode))

            # compute returns
            returns = self.compute_returns(episode)

            # update Q-values
            self.update_Q(returns)

            # optional epsilon decay
            if epsilon_decay is not None:
                epsilon = max(min_epsilon, epsilon * epsilon_decay)

            # optional progress logging
            if (episode_idx + 1) % 500 == 0:
                avg_len = sum(episode_lengths[-500:]) / 500
                print(
                    f"Episode {episode_idx + 1}/{num_episodes} | "
                    f"ε={epsilon:.3f} | "
                    f"Avg length (last 500): {avg_len:.1f}"
                )

        return episode_lengths
