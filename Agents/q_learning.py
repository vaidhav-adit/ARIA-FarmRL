import numpy as np  # Provides vectorized math + RNG for ε-greedy exploration.


class QLearningAgent:  # Tabular Q-learning with uniform binning of each continuous state dimension.
    N_BINS = 5  # Number of interval buckets per state feature (5^8 combinations possible, but sparse dict storage).
    N_ACTIONS = 2  # Two legal irrigation choices encoded as integers {0,1}.

    def __init__(  # Hyper-parameters mirror classical Watkins–Dayan Q-learning.
        self,  # Instance being configured.
        alpha: float = 0.1,  # TD learning rate scaling each backup magnitude.
        gamma: float = 0.99,  # Discount factor for future returns (close to 1 for long horizons).
        epsilon: float = 1.0,  # Initial exploration probability for ε-greedy policy.
        epsilon_min: float = 0.05,  # Floor preventing premature convergence to pure exploitation.
        epsilon_decay: float = 0.995,  # Multiplicative cooling factor applied after every episode.
    ):
        self.alpha = alpha  # Store learning rate on the instance.
        self.gamma = gamma  # Store discount factor.
        self.epsilon = epsilon  # Mutable exploration rate decayed over training.
        self.epsilon_min = epsilon_min  # Lower bound for ε after decay.
        self.epsilon_decay = epsilon_decay  # Per-episode multiplicative shrink factor.
        self.q_table: dict = {}  # Sparse map from discretized tuple → length-2 numpy Q vector.
        self.episode = 0  # Counter incremented in end_episode.
        self.total_updates = 0  # Number of TD backups applied (diagnostic only).

    def _discretize(self, state: np.ndarray) -> tuple:  # Convert float vector into hashable bin indices.
        bins = np.linspace(0, 1, self.N_BINS + 1)[1:-1]  # Interior thresholds splitting [0,1] into N_BINS segments.
        return tuple(int(np.digitize(s, bins)) for s in state)  # Per-dimension bin index list frozen as tuple key.

    def _get_q_values(self, discrete_state: tuple) -> np.ndarray:  # Fetch or lazily initialize Q-row for state.
        if discrete_state not in self.q_table:  # First time visiting this abstract state.
            self.q_table[discrete_state] = np.zeros(self.N_ACTIONS)  # Optimistic zeros (could switch if needed).
        return self.q_table[discrete_state]  # Return a mutable view into stored values.

    def select_action(self, state: np.ndarray) -> int:  # ε-greedy action sampling interface.
        if np.random.random() < self.epsilon:  # With probability ε explore uniformly.
            return np.random.randint(self.N_ACTIONS)  # Random legal irrigation decision.
        discrete = self._discretize(state)  # Map continuous observation to table coordinates.
        return int(np.argmax(self._get_q_values(discrete)))  # Greedy exploit with arbitrary tie-breaking by numpy.

    def update(  # Off-policy TD(0) backup toward max future value.
        self,  # Instance reference.
        state: np.ndarray,  # State before action.
        action: int,  # Action index executed.
        reward: float,  # Scalar reward from environment transition.
        next_state: np.ndarray,  # Successor observation.
        done: bool,  # True if episode terminated on this transition.
    ) -> None:  # Mutates Q-table in place; returns nothing.
        s = self._discretize(state)  # Discrete coordinates for current state.
        s_ = self._discretize(next_state)  # Discrete coordinates for next state.
        q_current = self._get_q_values(s)[action]  # Q(s,a) before update.
        if done:  # Terminal transition: no bootstrap term.
            td_target = reward  # Target equals immediate reward only.
        else:  # Non-terminal: bootstrap from largest action-value at next state.
            td_target = reward + self.gamma * np.max(self._get_q_values(s_))  # Off-policy greedy successor value.
        td_error = td_target - q_current  # Temporal difference residual.
        self.q_table[s][action] += self.alpha * td_error  # Watkins Q-learning update in place.
        self.total_updates += 1  # Housekeeping counter.

    def end_episode(self) -> None:  # Hook invoked once per episode by train.py.
        self.episode += 1  # Advance episode index.
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)  # Geometric ε decay with clamp.

    def get_stats(self) -> dict:  # Lightweight diagnostics for stdout logging.
        return {  # Immutable snapshot as plain dict for JSON compatibility.
            "episode": self.episode,  # How many episodes have finished.
            "epsilon": round(self.epsilon, 4),  # Exploration probability (rounded for display).
            "q_table_size": len(self.q_table),  # Number of visited abstract states.
            "total_updates": self.total_updates,  # Total TD steps.
        }
