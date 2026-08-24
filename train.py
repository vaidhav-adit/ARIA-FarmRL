# Train seven irrigation agents on the seasonal farm simulator and export learning curves to disk.


from __future__ import annotations  # Postpone evaluation of type hints referencing classes not yet defined.

import argparse  # Parse CLI flags such as episode count or random seed.
import json  # Serialize per-episode metrics for later plotting.
import os  # Locate this file and build output directories.
import sys  # Adjust import path so local modules resolve without installing a package.

import numpy as np  # Numerical averages for logging tail statistics.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Absolute path to the submission folder root.
sys.path.insert(0, SCRIPT_DIR)  # Ensure `import growth_curve` works when launched from anywhere.

from growth_curve import expected_growth  # Day-index target growth used to score final deviation.
from farm_env import FarmEnv  # Simulator implementing phased weather, soil physics, and reward terms.

from agents.q_learning import QLearningAgent  # Tabular ε-greedy off-policy control.
from agents.sarsa import SARSAAgent  # Tabular on-policy control with true next-action backups.
from agents.dqn import DQNAgent  # Neural fitted Q-iteration with replay buffer + target net.
from agents.reinforce import REINFORCEAgent  # Monte Carlo policy gradients with optional baseline net.
from agents.actor_critic import ActorCriticAgent  # Advantage actor–critic with entropy bonus.
from agents.ppo import PPOAgent  # Clipped surrogate policy optimization per episode.


class NumpyEncoder(json.JSONEncoder):  # Teach json.dumps how to handle numpy scalar objects.
    def default(self, obj):  # json.JSONEncoder hook invoked for non-primitive types.
        if isinstance(obj, np.floating):  # Any numpy float{16,32,64} should become Python float.
            return float(obj)  # Native JSON number type.
        if hasattr(obj, "item"):  # numpy scalar integer types expose .item() to cast down.
            return obj.item()  # Convert zero-dimensional arrays to pure Python scalars.
        return super().default(obj)  # Fall back to stock errors for unsupported objects.


def final_deviation(state: np.ndarray) -> float:  # Compute absolute gap between final biomass and agronomic target.
    final_growth = float(state[2])  # Third state component always stores growth score after the last transition.
    return round(abs(final_growth - expected_growth(90)), 4)  # Compare to interpolated expectation on harvest day.


def run_episode_generic(env: FarmEnv, agent) -> tuple[float, int, float]:  # Generic loop for most algorithms.
    state = env.reset()  # Fresh season with stochastic weather trajectory.
    total_reward = 0.0  # Accumulator for scalar reward each day.
    water = 0  # Count how often the pump action fires successfully or not — counts attempts? Actually counts action==1. Good.
    done = False  # Termination flag from environment.
    while not done:  # Exactly 90 iterations unless reset mid-way (should not happen).
        action = agent.select_action(state)  # Stochastic or ε-greedy policy evaluation + exploration.
        next_state, reward, done = env.step(action)  # Advance weather, soil, growth; receive feedback.
        agent.update(state, action, reward, next_state, done)  # Algorithm-specific learning update.
        if action == 1:  # Count timesteps where the agent commanded irrigation (action index 1).
            water += 1  # Running tally used for the water-use learning curve in compare.py.
        total_reward += reward  # Episode return for logging.
        state = next_state  # Roll forward for the next day.
    agent.end_episode()  # Decay ε, sync target nets, or run policy-gradient summaries.
    return round(total_reward, 4), water, final_deviation(state)  # Pack stats for JSON serialization.


def run_episode_sarsa(env: FarmEnv, agent: SARSAAgent) -> tuple[float, int, float]:  # SARSA needs paired actions.
    state = env.reset()  # Initial observation at dawn of day zero.
    action = agent.select_action(state)  # Choose first action before the transition loop begins.
    total_reward = 0.0  # Running sum identical to generic runner semantics.
    water = 0  # Pump counter for plotting.
    while True:  # Manual exit on terminal transition to align Q(s',a') backups cleanly.
        next_state, reward, done = env.step(action)  # Environment evolves using the committed action.
        total_reward += reward  # Accumulate return.
        if action == 1:  # Irrigation command on this day.
            water += 1  # Accumulate attempts for plotting against episodes.
        if done:  # No next action exists once the season ends.
            agent.update(state, action, reward, next_state, True, None)  # TD target reduces to immediate reward only.
            break  # Exit the perpetual loop safely.
        next_action = agent.select_action(next_state)  # On-policy successor action required by SARSA math.
        agent.update(state, action, reward, next_state, False, next_action)  # Full TD(0) backup with Q(s',a').
        state = next_state  # Advance time index.
        action = next_action  # Carry the already sampled next action into the next iteration without re-drawing.
    agent.end_episode()  # ε-decay identical to other tabular agents.
    return round(total_reward, 4), water, final_deviation(state)  # Tuple mirroring generic runner contract.


def train_one(  # Train a single agent for many episodes and flush metrics to JSON.
    name: str,  # Human-readable agent label (also drives output filename).
    agent,  # Concrete instance obeying select_action/update/end_episode protocol.
    episodes: int,  # Number of full seasons to simulate.
    seed_base: int,  # Integer offset so successive episodes diverge via FarmEnv(seed=...).
    results_dir: str,  # Destination directory path for the JSON artifact.
) -> dict:
    print(f"\n{'=' * 50}\nTraining: {name}\n{'=' * 50}")  # Section banner in the console transcript.
    rewards: list[float] = []  # Per-episode totals for reward subplot.
    water_log: list[int] = []  # Parallel list of pump counts for water subplot.
    deviations: list[float] = []  # Final |growth−expected| samples for deviation subplot.

    for ep in range(1, episodes + 1):  # Episodes numbered from 1 for human-friendly logs.
        env = FarmEnv(seed=seed_base + ep)  # Independent weather trajectory per episode via RNG seed.
        if name == "SARSA":  # Branch only where backup action semantics matter.
            r, w, d = run_episode_sarsa(env, agent)  # SARSA-specific trajectory roll-out.
        else:  # Every other agent shares the generic stepping contract.
            r, w, d = run_episode_generic(env, agent)  # Standard interface.
        rewards.append(r)  # Archive scalar return.
        water_log.append(w)  # Archive actuator usage statistic.
        deviations.append(d)  # Archive agronomic error.

        log_stride = max(1, episodes // 10)  # Aim for ~10 progress lines regardless of total episode budget.
        if ep % log_stride == 0 or ep == episodes:  # Always print the final episode exactly once.
            k = min(50, ep)  # Rolling window width for quick moving averages in stdout.
            avg_r = round(float(np.mean(rewards[-k:])), 4)  # Mean recent return.
            avg_w = round(float(np.mean(water_log[-k:])), 1)  # Mean irrigation count.
            avg_d = round(float(np.mean(deviations[-k:])), 4)  # Mean terminal deviation.
            stats = agent.get_stats()  # Algorithm-specific diagnostics dict.
            eps = stats.get("epsilon", stats.get("variant", "-"))  # Prefer ε for tabular agents else variant tag.
            print(  # Compact progress line mirrors compare.py smoothing only qualitatively.
                f"  Ep {ep:>4} | avg_reward={avg_r:>8.4f} | "
                f"avg_water={avg_w:>4.1f} | avg_dev={avg_d:.4f} | stats={eps}"
            )

    out = {  # Payload serialized verbatim into JSON for plotting + archival.
        "agent": name,  # Pretty label.
        "episodes": episodes,  # Metadata for downstream tools validating array lengths.
        "rewards": rewards,  # List[float] length == episodes.
        "water": water_log,  # Parallel list of ints.
        "deviations": deviations,  # Parallel list of floats.
        "final_stats": agent.get_stats(),  # Snapshot of internal counters after the last update.
    }
    fname = name.lower().replace(" ", "_") + ".json"  # Lowercase filenames stay shell-friendly.
    path = os.path.join(results_dir, fname)  # results/<agent>.json
    with open(path, "w", encoding="utf-8") as f:  # UTF-8 keeps international text safe.
        json.dump(out, f, indent=2, cls=NumpyEncoder)  # Pretty-print with numpy-safe encoder.
    print(f"  Saved -> {path}")  # Echo disk location for quick inspection.
    return out  # Allow caller to aggregate summaries without rereading JSON.


def main() -> None:  # CLI entry point invoked by `python train.py`.
    ap = argparse.ArgumentParser(description="Train irrigation RL agents on FarmEnv")  # Help text in `--help`.
    ap.add_argument("--episodes", type=int, default=500, help="Seasons per agent")  # Default episode budget per learner.
    ap.add_argument("--seed", type=int, default=0, help="Base RNG seed for environment sampling")  # Reproducibility knob.
    ap.add_argument(  # Plotting can be heavy in CI so expose a switch.
        "--no-plot", action="store_true", help="Skip generating results/comparison.png after training"
    )
    args = ap.parse_args()  # Materialize Namespace with user overrides.

    results_dir = os.path.join(SCRIPT_DIR, "results")  # Central store for JSON + PNG artifacts.
    os.makedirs(results_dir, exist_ok=True)  # Idempotent directory creation.

    print("Farm RL — training on phased seasonal simulator")  # Banner clarifying code path.
    print(f"Episodes per agent: {args.episodes}\n")  # Confirm workload before long runs.

    agents: list[tuple[str, object]] = [  # Instantiate fresh learners in presentation order.
        ("Q-Learning", QLearningAgent()),  # Sparse tabular Q.
        ("SARSA", SARSAAgent()),  # On-policy tabular Q.
        ("DQN", DQNAgent()),  # Deep Q-learning.
        ("REINFORCE-vanilla", REINFORCEAgent(use_baseline=False)),  # Pure policy gradient.
        ("REINFORCE-baseline", REINFORCEAgent(use_baseline=True)),  # Variance-reduced variant.
        ("Actor-Critic", ActorCriticAgent()),  # A2C-style online updates.
        ("PPO", PPOAgent()),  # Clipped PPO epochs.
    ]

    all_results: list[dict] = []  # Collect dicts for terminal summary without re-reading disk.
    for name, ag in agents:  # Train sequentially to limit RAM (each net stays resident until next begins).
        all_results.append(train_one(name, ag, args.episodes, args.seed, results_dir))  # Blocking call per agent.

    tail = min(100, args.episodes)  # Tail average window cannot exceed available history length.
    print(f"\n{'=' * 70}")  # Wider rule separating aggregate table.
    print(f"{'Agent':<22} {'Avg reward':>12} {'Avg water':>10} {'Avg dev':>10}")  # Column headings.
    print("-" * 70)  # Divider line.
    for r in all_results:  # One row per trained agent.
        avg_r = round(float(np.mean(r["rewards"][-tail:])), 4)  # Mean late-episode return.
        avg_w = round(float(np.mean(r["water"][-tail:])), 1)  # Mean late-episode irrigation count.
        avg_d = round(float(np.mean(r["deviations"][-tail:])), 4)  # Mean late-episode deviation.
        print(f"{r['agent']:<22} {avg_r:>12.4f} {avg_w:>10.1f} {avg_d:>10.4f}")  # Fixed-width row.
    print("=" * 70)  # Closing rule.
    print(f"\nJSON logs: {results_dir}/")  # Remind where raw time-series live.

    if not args.no_plot:  # Optionally skip matplotlib entirely.
        import compare  # Deferred import keeps `--help` fast if matplotlib missing at import time.

        compare.plot_all(SCRIPT_DIR)  # Build the three-panel figure beside the JSON files.


if __name__ == "__main__":  # Standard Python idiom guarding CLI execution.
    main()  # Dispatch to argument parsing + training loop.
