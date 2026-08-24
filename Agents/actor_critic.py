import numpy as np  # Numerical helpers + stats for logging average losses.
import torch  # Core tensor library with autograd.
import torch.nn as nn  # Neural layers / modules.
import torch.optim as optim  # Adam optimization.
import torch.nn.functional as F  # Functional API (mse_loss, etc.) without extra module state.


class ActorCriticNet(nn.Module):  # Shared trunk feeding separate policy logits and state-value head.
    def __init__(self, state_dim: int = 8, action_dim: int = 2, hidden: int = 64):  # Default widths consistent across agents.
        super().__init__()  # nn.Module base constructor.
        self.trunk = nn.Sequential(  # Shared nonlinear feature extractor ψ_θ(s).
            nn.Linear(state_dim, hidden),  # First projection expanding raw inputs.
            nn.SiLU(),  # Smooth activation in trunk for stability.
            nn.Linear(hidden, hidden),  # Mix features in latent space.
            nn.SiLU(),  # Second activation in trunk.
        )
        self.pi = nn.Linear(hidden, action_dim)  # Policy head outputting unnormalized logits per action.
        self.v = nn.Linear(hidden, 1)  # Value head outputting scalar V(s).

    def forward(self, x: torch.Tensor):  # Single forward computes both logits and baseline value.
        h = self.trunk(x)  # Shared representation h = ψ_θ(s).
        logits = self.pi(h)  # Linear map to action logits (pre-softmax).
        value = self.v(h).squeeze(-1)  # Remove singleton dimension yielding shape [batch].
        return logits, value  # Tuple consumed by downstream training code.


class ActorCriticAgent:  # One-step advantage actor–critic with entropy regularization bonus.
    def __init__(  # Hyper-parameters balance actor gradient scale vs critic MSE vs entropy exploration pressure.
        self,  # Instance under construction.
        state_dim: int = 8,  # Observation dimension.
        action_dim: int = 2,  # Discrete binary action space.
        lr: float = 3e-4,  # Adam learning rate on shared parameters θ.
        gamma: float = 0.99,  # Discount on bootstrapped value for non-terminal transitions.
        entropy_coef: float = 0.01,  # Entropy bonus coefficient encouraging diverse actions early on.
    ):
        self.gamma = gamma  # Store Bellman discount factor.
        self.entropy_coef = entropy_coef  # Trade-off knob for exploration bonus.
        self.action_dim = action_dim  # Unused directly but kept for future extensions.
        self.net = ActorCriticNet(state_dim, action_dim)  # Combined actor-critic parameter vector.
        self.opt = optim.Adam(self.net.parameters(), lr=lr)  # Single optimizer touching both heads + trunk.
        self.episode = 0  # Episodes completed counter.
        self.total_updates = 0  # Per-step gradient updates executed.
        self.losses: list[float] = []  # Scalar total loss history for diagnostics.

    def select_action(self, state: np.ndarray) -> int:  # Stochastic policy sampling without storing intermediate tensors.
        with torch.no_grad():  # Action selection during data collection stays inference-only for speed.
            logits, _ = self.net(torch.FloatTensor(state).unsqueeze(0))  # Forward with batch dimension B=1.
            dist = torch.distributions.Categorical(logits=logits)  # Categorical built from raw logits (numerically stable path).
            a = dist.sample()  # Sample discrete index.
            return int(a.item())  # Return Python int for env.step().

    def update(  # Online TD(0)-style update after each environment step (eligible trace length = 1).
        self,  # Bound instance.
        state: np.ndarray,  # s_t before transition.
        action: int,  # a_t executed.
        reward: float,  # r_t received.
        next_state: np.ndarray,  # s_{t+1} observed.
        done: bool,  # Whether s_{t+1} is terminal absorbing state.
    ):
        s = torch.FloatTensor(state).unsqueeze(0)  # Batch tensor [1, state_dim] still used in autograd path.
        ns = torch.FloatTensor(next_state).unsqueeze(0)  # Batch tensor for successor evaluation under no_grad pieces.
        a = torch.LongTensor([action])  # Long dtype required by Categorical.log_prob expectations.

        logits, value = self.net(s)  # Value is V(s_t); logits define π(a|s_t).
        with torch.no_grad():  # Bootstrap target treats next state value as constant label source.
            _, next_value = self.net(ns)  # Evaluate V(s_{t+1}) without building unnecessary graph through next logits for target.
            nv = float(next_value.item()) if not done else 0.0  # Zero bootstrap when episode ends (Monte Carlo cut).
        td_target = torch.tensor([reward + self.gamma * nv], dtype=torch.float32, device=value.device)  # Bellman target y = r + γ V(s').

        dist = torch.distributions.Categorical(logits=logits)  # Policy distribution π_θ(·|s_t).
        logp = dist.log_prob(a)  # Log π(a_t|s_t) with shape [1] retaining graph for policy loss.
        advantage = td_target - value  # Advantage A = TD target minus current value estimate (can be viewed as one-step TD error).

        actor_loss = -(logp * advantage.detach()).mean()  # Policy gradient loss uses detached advantages as coefficients.
        critic_loss = F.mse_loss(value, td_target)  # regress V(s) toward one-step return target.
        entropy = dist.entropy().mean()  # Average policy entropy; subtracting it with positive coef adds exploration pressure.
        loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy  # Total loss: actor + scaled critic − entropy bonus.

        self.opt.zero_grad()  # Reset accumulated gradients on all shared parameters.
        loss.backward()  # Autograd computes ∂loss/∂θ across actor + critic pathways.
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)  # Stabilize updates under noisy farm rewards.
        self.opt.step()  # Adam applies the gradient step.

        self.losses.append(float(loss.item()))  # Track batch-less scalar loss for logging.
        self.total_updates += 1  # Increment per-step update count.

    def end_episode(self):  # Minimal hook: only increments episode counter (entropy/ε schedules could go here).
        self.episode += 1  # Maintain diagnostic parity with other agents.

    def get_stats(self) -> dict:  # Serialize short stats snapshot for train.py logging.
        return {  # Plain dict with only serializable Python scalars.
            "episode": self.episode,  # Episodes completed.
            "total_updates": self.total_updates,  # Number of gradient steps.
            "avg_loss": round(float(np.mean(self.losses[-100:])), 6) if self.losses else 0.0,  # Moving average diagnostic.
        }
