import random  # Uniformly sample minibatch indices without replacement from replay memory.
from collections import deque  # Fixed-capacity FIFO queue dropping oldest experiences automatically.

import numpy as np  # Bridge NumPy environment arrays → Torch tensors.
import torch  # Tensor computation + autograd engine.
import torch.nn as nn  # Neural building blocks (Linear, activations, Sequential containers).
import torch.optim as optim  # Optimization algorithms (Adam here).


class QNetwork(nn.Module):  # Small MLP representing state-action values for the binary irrigation MDP.
    def __init__(self, state_dim: int = 8, action_dim: int = 2, hidden: int = 64):  # Widths tuned for 8-D states.
        super().__init__()  # Register submodule parameters with nn.Module infrastructure.
        self.net = nn.Sequential(  # Ordered list of layers evaluated sequentially on each forward call.
            nn.Linear(state_dim, hidden),  # Project raw state vector into hidden feature space.
            nn.SiLU(),  # Smooth activation balancing ReLU-like gating with differentiability everywhere.
            nn.Linear(hidden, hidden),  # Second hidden transformation mixing learned features.
            nn.SiLU(),  # Nonlinearity after the second linear map.
            nn.Linear(hidden, action_dim),  # Emit one scalar Q(s,a) per discrete action in a single batched matmul.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Implements Q_θ(s).
        return self.net(x)  # Broadcasts over batch dimension if x.shape[0] > 1.


class ReplayBuffer:  # Experience replay storing decorrelated transitions for stable supervised Q targets.
    def __init__(self, capacity: int = 50_000):  # Capacity chosen to cover weeks of simulated data.
        self.buffer = deque(maxlen=capacity)  # Ring buffer automatically evicts stale tuples when full.

    def push(self, state, action, reward, next_state, done):  # Insert one transition tuple at the right end.
        self.buffer.append((state, action, reward, next_state, done))  # Tuple packing keeps fields aligned.

    def sample(self, batch_size: int):  # Draw a minibatch uniformly at random without replacement.
        batch = random.sample(self.buffer, batch_size)  # Python list of transition tuples length == batch_size.
        states, actions, rewards, next_states, dones = zip(*batch)  # Unzip into five parallel tuples.
        return (  # Return Torch tensors ready for GPU/CPU training code.
            torch.FloatTensor(np.array(states)),  # Shape [B, state_dim].
            torch.LongTensor(actions),  # Shape [B] integer indices in {0,1}.
            torch.FloatTensor(rewards),  # Shape [B] scalar rewards.
            torch.FloatTensor(np.array(next_states)),  # Shape [B, state_dim].
            torch.FloatTensor(dones),  # Shape [B] float masks {0.0,1.0}.
        )

    def __len__(self):  # Python protocol enabling `len(buffer)` syntactic sugar.
        return len(self.buffer)  # Current number of stored transitions (≤ capacity).


class DQNAgent:  # Double-network deep Q-learning with ε-greedy exploration and periodic target sync.
    def __init__(  # Hyper-parameters mirror Mnih et al. style defaults scaled to this sim.
        self,  # Instance being constructed.
        state_dim: int = 8,  # Observation dimensionality from FarmEnv.
        action_dim: int = 2,  # Binary decision problem.
        lr: float = 1e-3,  # Adam learning rate for Q-network weights.
        gamma: float = 0.99,  # Discount factor reward horizon ~ hundred days.
        epsilon: float = 1.0,  # Initial exploration probability for ε-greedy behavior.
        epsilon_min: float = 0.05,  # Floor ε never decays below for continual mild exploration.
        epsilon_decay: float = 0.995,  # Multiplicative cooling factor applied once per episode.
        batch_size: int = 64,  # Number of replay tuples per gradient step.
        target_update: int = 10,  # How many episodes between Polyak/full target network copies.
    ):
        self.action_dim = action_dim  # Cache action space cardinality for argmax and random exploration.
        self.gamma = gamma  # Store discount for TD targets.
        self.epsilon = epsilon  # Mutable exploration rate.
        self.epsilon_min = epsilon_min  # Exploration floor hyper-parameter.
        self.epsilon_decay = epsilon_decay  # Episode-wise exponential decay factor on ε.
        self.batch_size = batch_size  # Minibatch cardinality hyper-parameter.
        self.target_update = target_update  # Episode stride for syncing target weights.

        self.q_net = QNetwork(state_dim, action_dim)  # Online network optimized every gradient step.
        self.t_net = QNetwork(state_dim, action_dim)  # Target network evaluated with no_grad for stable targets.
        self.t_net.load_state_dict(self.q_net.state_dict())  # Start target identical to online for coherent bootstrap.
        self.t_net.eval()  # Disable dropout/batch-norm training behaviors (none here but conventional).

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)  # Adam tracks per-parameter moments.
        self.loss_fn = nn.MSELoss()  # Mean squared TD error between prediction Q(s,a) and one-step target y.

        self.buffer = ReplayBuffer()  # Experience replay ring buffer instance.

        self.episode = 0  # Counter used both for ε schedule and target refresh cadence.
        self.total_updates = 0  # Gradient steps performed (diagnostic only).
        self.losses: list[float] = []  # Rolling log of batch MSE losses for stdout statistics.

    def select_action(self, state: np.ndarray) -> int:  # ε-greedy interface exposed to environment driver.
        if np.random.random() < self.epsilon:  # Explore branch.
            return np.random.randint(self.action_dim)  # Uniform random legal action.
        with torch.no_grad():  # Inference-only path saves memory and compute.
            s = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension → shape [1,8].
            q = self.q_net(s)  # Forward pass yields shape [1,2] Q estimates.
            return int(q.argmax().item())  # Break ties by smallest index (numpy/torch default).

    def update(  # One environment step may trigger zero or one gradient update depending on buffer fill.
        self,  # Bound method.
        state: np.ndarray,  # s_t.
        action: int,  # a_t.
        reward: float,  # r_t.
        next_state: np.ndarray,  # s_{t+1}.
        done: bool,  # Episode terminal flag.
    ):
        self.buffer.push(state, action, reward, next_state, float(done))  # Store float mask for tensor math.
        if len(self.buffer) < self.batch_size:  # Not enough data yet to form a minibatch.
            return  # Skip training early in episode one.

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)  # Random minibatch tensors.
        q_current = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)  # Select taken-action columns only.

        with torch.no_grad():  # Target values treated as regression labels without backprop through target net.
            q_next = self.t_net(next_states).max(1)[0]  # max_a' Q_target(s',a').
            q_target = rewards + self.gamma * q_next * (1 - dones)  # Bellman backup with terminal mask.

        loss = self.loss_fn(q_current, q_target)  # Scalar MSE summed across batch.
        self.optimizer.zero_grad()  # Clear stale gradients from previous minibatches.
        loss.backward()  # Autodiff computes ∂loss/∂θ for online network parameters θ.
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)  # Global norm clipping prevents explosion.
        self.optimizer.step()  # Apply Adam step using freshly computed gradients.

        self.losses.append(loss.item())  # Float snapshot for logging moving averages.
        self.total_updates += 1  # Increment gradient step counter.

    def end_episode(self):  # Hook called once per episode after final transition processed.
        self.episode += 1  # Advance episode counter for ε schedule + target sync period.
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)  # Exponential explore decay with floor.
        if self.episode % self.target_update == 0:  # Periodic hard update keeps targets slowly chasing online net.
            self.t_net.load_state_dict(self.q_net.state_dict())  # Copy all matching parameter tensors.

    def get_stats(self) -> dict:  # Lightweight training diagnostics for console logs / JSON metadata.
        avg_loss = round(float(np.mean(self.losses[-100:])), 6) if self.losses else 0.0  # Mean over last ≤100 steps.
        return {  # Dictionary merged into train.py outputs.
            "episode": self.episode,  # Completed episode counter.
            "epsilon": round(self.epsilon, 4),  # Exploration probability rounded for humans.
            "buffer_size": len(self.buffer),  # How many transitions currently stored.
            "total_updates": self.total_updates,  # Gradient steps taken so far.
            "avg_loss": avg_loss,  # Recent MSE average for debugging stability.
        }

    def save(self, path: str):  # Optional checkpointing hook (train.py may call if extended).
        torch.save(self.q_net.state_dict(), path)  # Persist only online weights for inference deployments.
