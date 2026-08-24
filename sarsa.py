import numpy as np  # Random draws + array math for SARSA backups.


class SARSAAgent:  # Tabular on-policy control learning Q^π instead of Q^*.
    N_BINS = 5  # Same discretization resolution as Q-learning for fairness.
    N_ACTIONS = 2  # Environment supports binary decisions only.

    def __init__(  # Hyper-parameters parallel the companion Q-learning agent.
        self,  # Instance under construction.
        alpha: float = 0.1,  # Learning rate for TD error.
        gamma: float = 0.99,  # Discount on future rewards within the Bellman backup.
        epsilon: float = 1.0,  # Starting exploration rate.
        epsilon_min: float = 0.05,  # Minimum ε after decay schedule saturates.
        epsilon_decay: float = 0.995,  # Per-episode multiplicative cooling.
    ):
        self.alpha = alpha  # Stash learning rate.
        self.gamma = gamma  # Stash discount factor.
        self.epsilon = epsilon  # Mutable ε tracker.
        self.epsilon_min = epsilon_min  # Lower exploration bound.
        self.epsilon_decay = epsilon_decay  # Decay multiplier applied once per episode.
        self.q_table: dict = {}  # Sparse Q storage keyed by discretized tuple.
        self.episode = 0  # Episode counter for logging.
        self.total_updates = 0  # Count TD backups performed.

    def _discretize(self, state: np.ndarray) -> tuple:  # Identical binning strategy as Q-learning.
        bins = np.linspace(0, 1, self.N_BINS + 1)[1:-1]  # Internal bin edges excluding endpoints.
        return tuple(int(np.digitize(s, bins)) for s in state)  # Tuple of nonnegative bin indices.

    def _get_q_values(self, discrete_state: tuple) -> np.ndarray:  # Lazy initialization helper.
        if discrete_state not in self.q_table:  # Allocate row on first visit.
            self.q_table[discrete_state] = np.zeros(self.N_ACTIONS)  # Start neutral.
        return self.q_table[discrete_state]  # Return pointer to the two Q entries.

    def select_action(self, state: np.ndarray) -> int:  # ε-greedy behavior policy shared with Q-learning structure.
        if np.random.random() < self.epsilon:  # Random exploration branch.
            return np.random.randint(self.N_ACTIONS)  # Uniform over {0,1}.
        discrete = self._discretize(state)  # Binned coordinates.
        return int(np.argmax(self._get_q_values(discrete)))  # Greedy exploitation.

    def update(  # SARSA TD target references actual next action a', not max_a Q(s',a).
        self,  # Instance bound method.
        state: np.ndarray,  # Starting observation.
        action: int,  # Executed action a_t.
        reward: float,  # Observed reward r_t.
        next_state: np.ndarray,  # s_{t+1}.
        done: bool,  # Episode terminal flag.
        next_action: int | None,  # a_{t+1} sampled from policy unless terminal.
    ) -> None:
        s = self._discretize(state)  # Hashable key for current state.
        s_ = self._discretize(next_state)  # Hashable key for successor state.
        q_current = self._get_q_values(s)[action]  # Q(s,a) before backup.
        if done:  # If transition ended episode, future term omitted.
            td_target = reward  # Only immediate reward contributes.
        else:  # Non-terminal transitions need on-policy successor action value.
            assert next_action is not None  # train.py contract: must supply paired action.
            td_target = reward + self.gamma * self._get_q_values(s_)[next_action]  # SARSA bootstrap uses Q(s',a').
        td_error = td_target - q_current  # TD residual δ.
        self.q_table[s][action] += self.alpha * td_error  # In-place SARSA update.
        self.total_updates += 1  # Increment diagnostics counter.

    def end_episode(self) -> None:  # Episode boundary hook for exploration annealing.
        self.episode += 1  # Increment finished episode count.
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)  # Standard ε schedule.

    def get_stats(self) -> dict:  # Return JSON-friendly diagnostics snapshot.
        return {  # Keys mirror Q-learning for uniform logging in train.py.
            "episode": self.episode,  # Number of completed episodes.
            "epsilon": round(self.epsilon, 4),  # Current exploration probability (rounded).
            "q_table_size": len(self.q_table),  # Number of visited abstract states.
            "total_updates": self.total_updates,  # TD steps executed.
        }
