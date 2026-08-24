import numpy as np  # Array packaging of episode trajectories before Torch tensor creation.
import torch  # GPU/CPU tensors + autograd for policy-gradient optimization.
import torch.nn as nn  # Module API for policy and baseline networks.
import torch.optim as optim  # Adam optimizers for each parameter group.


class PolicyNetwork(nn.Module):  # Differentiable softmax policy π_θ(a|s).
    def __init__(self, state_dim: int = 8, action_dim: int = 2, hidden: int = 64):  # Architectural defaults.
        super().__init__()  # Register parameters/submodules with PyTorch.
        self.net = nn.Sequential(  # Standard feed-forward mapping state → logits → softmax handled outside.
            nn.Linear(state_dim, hidden),  # First fully connected layer expands state features.
            nn.SiLU(),  # Smooth nonlinearity aiding optimization in continuous control-inspired tasks.
            nn.Linear(hidden, hidden),  # Second layer mixes hidden units nonlinearly.
            nn.SiLU(),  # Second activation.
            nn.Linear(hidden, action_dim),  # Emit one logit per discrete action candidate.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Evaluate π(a|s) as a categorical distribution parameter.
        return torch.softmax(self.net(x), dim=-1)  # Ensure probabilities sum to 1 along action dimension.


class ValueNetwork(nn.Module):  # Learned baseline predicting expected return V(s) from features of s alone.
    def __init__(self, state_dim: int = 8, hidden: int = 64):  # Baseline capacity comparable to policy trunk.
        super().__init__()  # nn.Module initialization.
        self.net = nn.Sequential(  # Scalar regression network.
            nn.Linear(state_dim, hidden),  # Map states into latent vector.
            nn.SiLU(),  # Nonlinearity.
            nn.Linear(hidden, hidden),  # Additional depth for expressive value function.
            nn.SiLU(),  # Activation after second linear.
            nn.Linear(hidden, 1),  # Collapse hidden vector to scalar value estimate.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Predict V(s) with shape suitable for broadcasting.
        return self.net(x).squeeze(-1)  # Remove trailing singleton dimension to simplify loss formulas.


class REINFORCEAgent:  # Implements episodic policy gradients with optional variance-reducing baseline network.
    def __init__(  # Construction mirrors hyper-parameter choices used for benchmarking.
        self,  # Instance reference.
        state_dim: int = 8,  # Environment observation dimensionality.
        action_dim: int = 2,  # Two discrete irrigation actions.
        lr_policy: float = 1e-3,  # Policy Adam learning rate.
        lr_value: float = 1e-3,  # Baseline Adam learning rate when baseline enabled.
        gamma: float = 0.99,  # Discount factor weighting future rewards inside Monte Carlo returns.
        use_baseline: bool = True,  # If False, revert to classical REINFORCE with high-variance updates.
    ):
        self.gamma = gamma  # Save discount for end_episode return recursion.
        self.use_baseline = use_baseline  # Choose between two loss branches later.
        self.action_dim = action_dim  # Inform categorical sampling helper logic.

        self.policy = PolicyNetwork(state_dim, action_dim)  # Instantiate policy π_θ.
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=lr_policy)  # Optimizer tracks only policy weights.

        if use_baseline:  # Conditional baseline allocation to save parameters when unused.
            self.value = ValueNetwork(state_dim)  # Baseline value approximator V_φ(s).
            self.value_opt = optim.Adam(self.value.parameters(), lr=lr_value)  # Separate Adam state for φ.
        else:  # Baseline-free algorithm variant.
            self.value = None  # Explicit None eases debugging if accidentally referenced.

        self.states: list = []  # Trajectory memory for value regression (also harmless in no-baseline mode).
        self.rewards: list = []  # Rewards r_t collected for Monte Carlo target construction.
        self.log_probs: list = []  # Log-prob tensors retaining computation graph hooks until backward().

        self.episode = 0  # Counter of finished episodes.
        self.total_updates = 0  # Count of policy-gradient optimization steps.
        self.policy_losses: list[float] = []  # Diagnostic list of recent scalar losses.

    def select_action(self, state: np.ndarray) -> int:  # Sample stochastic action and stash log π for training.
        s = torch.FloatTensor(state).unsqueeze(0)  # Convert np.ndarray → tensor with batch dimension B=1.
        probs = self.policy(s)  # Forward pass yields π(a|s) row vector.
        m = torch.distributions.Categorical(probs)  # Build categorical distribution directly from softmax probs.
        action = m.sample()  # Draw integer action.
        self.log_probs.append(m.log_prob(action))  # Store tensor retaining graph for policy_loss.backward().
        return int(action.item())  # Environment API expects Python int not 0-dim tensor.

    def update(  # During the episode we only buffer transitions; updates occur in end_episode().
        self,  # Method binding.
        state: np.ndarray,  # s_t copied for baseline learning after rollout completes.
        action: int,  # Not used here because log-prob already captured during select_action().
        reward: float,  # Scalar reward appended for return computation.
        next_state: np.ndarray,  # Unused in pure Monte Carlo full-horizon REINFORCE.
        done: bool,  # Unused because rewards list length defines horizon when training fires.
    ):
        self.states.append(state)  # Append numpy observation for optional baseline regression.
        self.rewards.append(reward)  # Append immediate reward for discounted return recursion.

    def end_episode(self):  # Complete policy / value optimization pass once trajectory ends.
        self.episode += 1  # Increment episode counter for parity with other agents’ logs.
        G = 0.0  # Scalar accumulator for reverse discounted return.
        returns: list[float] = []  # Chronological list of G_t values after we reverse iterate rewards.
        for r in reversed(self.rewards):  # Start from terminal reward and walk backward through time.
            G = r + self.gamma * G  # Standard discounted Monte Carlo return recursion.
            returns.insert(0, G)  # Maintain time order by inserting at front.

        returns_tensor = torch.FloatTensor(returns)  # FloatTensor [T] for vectorized loss math.
        if len(returns) > 1:  # Guard std-dev normalization against empty or length-1 episodes.
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)  # Normalize advantages scale-free.

        log_probs_tensor = torch.stack(self.log_probs)  # Concatenate per-step log-prob tensors along time.

        if self.use_baseline:  # Actor with learned critic-style baseline but separate networks.
            states_tensor = torch.FloatTensor(np.array(self.states))  # Stack observations along new axis 0.
            values = self.value(states_tensor)  # Predict V(s_t) for every timestep in trajectory.
            advantages = returns_tensor - values.detach()  # Stop-gradient through baseline when policy differentiates.
            policy_loss = -(log_probs_tensor * advantages).mean()  # Policy ascent along advantage-weighted log π.
            value_loss = nn.MSELoss()(values, returns_tensor)  # Baseline regresses empirical returns to reduce variance of policy gradient estimator.

            self.policy_opt.zero_grad()  # Reset policy grads.
            policy_loss.backward()  # Differentiate policy_loss w.r.t. θ.
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)  # Clip global norm for stability.
            self.policy_opt.step()  # Adam update on θ.

            self.value_opt.zero_grad()  # Reset baseline grads.
            value_loss.backward()  # Differentiate quadratic loss w.r.t. φ.
            self.value_opt.step()  # Adam update on φ.

        else:  # Classical REINFORCE without subtracting learned baseline.
            policy_loss = -(log_probs_tensor * returns_tensor).mean()  # REINFORCE uses raw (possibly standardized) returns as weights.
            self.policy_opt.zero_grad()  # Clear gradients.
            policy_loss.backward()  # Backprop through log-prob chain.
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)  # Clip to avoid exploding logits.
            self.policy_opt.step()  # Apply Adam step.

        self.policy_losses.append(float(policy_loss.item()))  # Record float for diagnostic averaging.
        self.total_updates += 1  # Count completed policy-gradient updates.

        self.states = []  # Flush trajectory buffers for next episode.
        self.rewards = []  # Clear reward list.
        self.log_probs = []  # Clear log-prob list to avoid graph leakage across episodes.

    def get_stats(self) -> dict:  # Metrics surfaced to train.py for stdout/JSON.
        avg_loss = (  # Compute mean of last up-to-100 policy losses if available.
            round(float(np.mean(self.policy_losses[-100:])), 6) if self.policy_losses else 0.0
        )
        return {  # Serialize compact diagnostic dict.
            "episode": self.episode,  # Finished episode count.
            "total_updates": self.total_updates,  # Number of policy-gradient steps.
            "avg_policy_loss": avg_loss,  # Recent mean policy loss for debugging scale.
            "variant": "with_baseline" if self.use_baseline else "vanilla",  # Human-readable mode tag for logs.
        }
