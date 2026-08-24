import numpy as np  # Array stacking, shuffles, and Monte Carlo return accumulation helpers.
import torch  # PyTorch core.
import torch.nn as nn  # Neural modules + MSELoss instantiation.
import torch.optim as optim  # Adam optimizer driving PPO updates.


class PPONet(nn.Module):  # Shared-body architecture producing both action logits and scalar value estimates V(s).
    def __init__(self, state_dim: int = 8, action_dim: int = 2, hidden: int = 64):  # Reasonable default width for 8-D inputs.
        super().__init__()  # nn.Module registration.
        self.trunk = nn.Sequential(  # Shared nonlinear feature map ψ_θ(s).
            nn.Linear(state_dim, hidden),  # Lift observations to hidden dimension.
            nn.SiLU(),  # Smooth activation for gradient-friendly training.
            nn.Linear(hidden, hidden),  # Additional depth for representational capacity.
            nn.SiLU(),  # Second activation in trunk.
        )
        self.pi = nn.Linear(hidden, action_dim)  # Policy logits before softmax inside distributions.Categorical.
        self.v = nn.Linear(hidden, 1)  # Scalar value head per sample row.

    def forward(self, x: torch.Tensor):  # Evaluate both heads given batched states x.
        h = self.trunk(x)  # Shared hidden activations for each row in batch.
        return self.pi(h), self.v(h).squeeze(-1)  # Return logits [B, action_dim] and values [B].


class PPOAgent:  # On-policy actor–critic improved via clipped probability ratios + multi-epoch reuse of rollout data.
    def __init__(  # Hyper-parameters controlling stability/ sample efficiency trade-offs on 90-step episodes.
        self,  # Instance under construction.
        state_dim: int = 8,  # Farm observation dimension.
        action_dim: int = 2,  # Binary irrigation MDP.
        lr: float = 3e-4,  # Adam learning rate.
        gamma: float = 0.99,  # Discount used when reducing rewards to Monte Carlo returns along each path.
        eps_clip: float = 0.2,  # Clip parameter ε restricting policy update aggressiveness each epoch.
        epochs: int = 10,  # Full-data passes per episode after collecting 90-step trajectory.
        mini_batch: int = 32,  # Subsample size drawn without replacement inside each epoch pass.
        entropy_coef: float = 0.01,  # Entropy bonus coefficient for exploration pressure.
    ):
        self.gamma = gamma  # Store discount for GAE-less Monte Carlo return estimator used here (simple cumulant).
        self.eps_clip = eps_clip  # Clip range for PPO surrogate objective.
        self.epochs = epochs  # Number of optimization epochs over stored rollout per episode.
        self.mini_batch = mini_batch  # Minibatch cardinality for SGD noise.
        self.entropy_coef = entropy_coef  # Entropy term scaling in total loss.
        self.action_dim = action_dim  # Reserved for diagnostics / future action masking.

        self.net = PPONet(state_dim, action_dim)  # Combined actor–critic parameter block.
        self.opt = optim.Adam(self.net.parameters(), lr=lr)  # Single Adam instance for all parameters.

        self.states: list[np.ndarray] = []  # Trajectory states s_t copied as numpy for cheap storage.
        self.actions: list[int] = []  # Integer actions executed at each time.
        self.log_probs: list[torch.Tensor] = []  # Detached log π_old(a_t|s_t) for each time with old policy snapshot.
        self.rewards: list[float] = []  # Scalar rewards r_t.
        self.dones: list[bool] = []  # Episode termination flags (always False until final True in this episodic task).

        self.episode = 0  # Episodes completed counter.
        self.updates = 0  # Number of minibatch gradient steps taken (not necessarily equal to episodes).

    def select_action(self, state: np.ndarray) -> int:  # Stochastic rollout policy using current network weights.
        with torch.no_grad():  # Sampling does not need gradients through logits during data collection.
            s = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension B=1.
            logits, _ = self.net(s)  # Forward pass acquiring policy logits only (value unused here).
            dist = torch.distributions.Categorical(logits=logits)  # Categorical policy π_θ.
            a = dist.sample()  # Drawinteger action index.
        return int(a.item())  # Return Python int to environment driver.

    def update(  # Append one transition worth of on-policy data; gradients computed only in end_episode().
        self,  # Reference to agent instance.
        state: np.ndarray,  # s_t observation copied into buffer.
        action: int,  # a_t index executed by env.
        reward: float,  # r_t scalar.
        next_state: np.ndarray,  # s_{t+1} not directly used here because Monte Carlo returns reprocess rewards list.
        done: bool,  # Terminal mask stored for return bootstrap resets in rare multi-episode buffers (here mostly terminal at end).
    ):
        s = torch.FloatTensor(state).unsqueeze(0)  # Tensor view of current state for recomputing log-prob under policy graph.
        logits, _ = self.net(s)  # Evaluate logits with grad-enabled tensors because log_prob may be built (but we detach when storing).
        dist = torch.distributions.Categorical(logits=logits)  # Policy at timestep t used to evaluate a_t density.
        logp = dist.log_prob(torch.tensor([action], dtype=torch.long))  # Log π(a_t|s_t) with correct dtype for discrete dist.
        self.log_probs.append(logp.detach())  # Store OLD log-prob snapshot detached from future graph to stabilize ratios.
        self.states.append(state.copy())  # Store array copy to avoid mutation if env reuses buffers.
        self.actions.append(action)  # Integer action list parallel to states.
        self.rewards.append(reward)  # Reward parallel list.
        self.dones.append(done)  # Done parallel list supporting episodic return cut logic.

    def end_episode(self):  # Main PPO optimization routine—multiple epochs over trajectory minibatches.
        self.episode += 1  # Increment episode counter first for logging clarity.
        if len(self.states) == 0:  # Guard against accidental empty rollouts.
            self._clear()  # Ensure lists reset even on degenerate calls.
            return  # Nothing else to train.

        states_t = torch.FloatTensor(np.array(self.states))  # [T, state_dim] tensor of the whole rollout.
        actions_t = torch.LongTensor(self.actions)  # [T] int64 actions matching time indices.
        old_logp = torch.stack(self.log_probs).detach()  # [T] vector of behavior policy log probs frozen as "old policy".

        rewards = self.rewards  # Alias for readability during return accumulation.
        dones = self.dones  # Parallel terminal flags.
        R = 0.0  # Running return accumulator from the end (Monte Carlo sums here).
        returns: list[float] = []  # Will become length-T list aligned chronologically.
        for r, d in zip(reversed(rewards), reversed(dones)):  # Iterate backward over paired sequences.
            if d:  # If terminal flag encountered, zero future bootstrap contribution (future is zero by definition).
                R = 0.0  # Reset running sum at episode cut (rare mid-buffer except final True).
            R = r + self.gamma * R  # Standard discounted return recursion along reversed time.
            returns.insert(0, R)  # Rebuild chronological order by prepending.

        returns_t = torch.FloatTensor(returns)  # Float tensor [T] regression targets for value head.

        with torch.no_grad():  # Advantage estimates treat old value predictions as constants.
            _, values = self.net(states_t)  # Evaluate value predictions V(s_t) with current network before updates (old baseline).
            advantages = returns_t - values  # Simple full-return advantage estimate (no GAE for brevity).
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize advantages for stable learning.

        n = len(self.states)  # Trajectory length T (≈90 here).
        idx = np.arange(n)  # Index array [0..T-1] being reshuffled each epoch.

        for _ in range(self.epochs):  # Multiple optimization passes over identical data (core PPO idea).
            np.random.shuffle(idx)  # Random permutation of timesteps for stochastic minibatches.
            for start in range(0, n, self.mini_batch):  # Slide minibatch window along permutation.
                batch = idx[start : start + self.mini_batch]  # Indices for current minibatch (may be smaller at end).
                b_states = states_t[batch]  # Gather state minibatch rows.
                b_actions = actions_t[batch]  # Gather corresponding actions.
                b_old_logp = old_logp[batch]  # Old log-prob entries for ratio computation.
                b_returns = returns_t[batch]  # Target returns for critic regression on minibatch.
                b_adv = advantages[batch]  # Advantage slice aligned with minibatch.

                logits, vals = self.net(b_states)  # Current policy + value predictions for minibatch using updated weights each inner step.
                dist = torch.distributions.Categorical(logits=logits)  # Current categorical policy π_θ_new.
                logp = dist.log_prob(b_actions)  # Log π_new(a|s) for replayed actions.
                ratio = torch.exp(logp - b_old_logp)  # Probability ratio r_t(θ) = π_new/π_old in log space for numerical stability.

                surr1 = ratio * b_adv  # Unclipped surrogate term.
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * b_adv  # Clipped surrogate term.
                actor_loss = -torch.min(surr1, surr2).mean()  # Pessimistic ( conservative ) combination forms trust region.
                critic_loss = nn.MSELoss()(vals, b_returns)  # Value function regression toward Monte Carlo returns.
                entropy = dist.entropy().mean()  # Average policy entropy encourages exploration when subtracted with coef>0.

                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy  # Total PPO loss decomposition.
                self.opt.zero_grad()  # Clear parameter gradients before backward.
                loss.backward()  # Differentiate combined loss w.r.t. network weights.
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)  # Mild global gradient clipping for PPO stability.
                self.opt.step()  # Adam applies update on shared parameters.
                self.updates += 1  # Count minibatch updates executed.

        self._clear()  # Flush rollout buffers preparing for next episode collection.

    def _clear(self):  # Reset temporary trajectory storage lists to empty defaults.
        self.states = []  # Clear state list.
        self.actions = []  # Clear action list.
        self.log_probs = []  # Clear old log-prob list.
        self.rewards = []  # Clear rewards list.
        self.dones = []  # Clear terminal flags list.

    def get_stats(self) -> dict:  # Minimal diagnostic struct for train.py.
        return {"episode": self.episode, "ppo_updates": self.updates}  # Episode count + aggregate minibatch update count.
