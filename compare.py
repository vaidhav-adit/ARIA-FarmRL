# Load per-agent JSON logs and render a three-panel comparison figure (reward, water, deviation).


import json  # Deserialize training logs saved after each agent finishes.
import os  # Build filesystem paths in a portable way.

import matplotlib  # Backend selection must happen before pyplot import when using Agg.
matplotlib.use("Agg")  # Headless renderer so training scripts can save PNGs without a display.

import matplotlib.gridspec as gridspec  # Arrange subplots: one wide row on top, two cells below.
import matplotlib.pyplot as plt  # High-level plotting API.
import numpy as np  # Smoothing via convolution and numerical stats.

# Canonical filesystem stem for each agent’s JSON (matches train.py output naming).
AGENT_NAMES = [  # Order controls legend stacking; keep consistent for readability.
    "q-learning",  # Tabular off-policy baseline.
    "sarsa",  # Tabular on-policy baseline.
    "dqn",  # Deep Q-network with replay.
    "reinforce-vanilla",  # Monte Carlo policy gradient without value baseline.
    "reinforce-baseline",  # Policy gradient with learned baseline for variance reduction.
    "actor-critic",  # Single-step actor–critic updates sharing trunk features.
    "ppo",  # Proximal policy optimization with clipped objective.
]

# Distinct colors per series so seven curves remain separable when printed in grayscale or color.
COLORS = {  # Hex codes chosen for contrast on white backgrounds.
    "q-learning": "#378ADD",  # Blue.
    "sarsa": "#1D9E75",  # Green.
    "dqn": "#D85A30",  # Orange.
    "reinforce-vanilla": "#888780",  # Gray.
    "reinforce-baseline": "#7F77DD",  # Violet.
    "actor-critic": "#E69F00",  # Amber.
    "ppo": "#CC79A7",  # Magenta.
}

SMOOTH_WIN = 20  # Rolling-mean window length (episodes) to suppress high-frequency noise in curves.


def smooth(data, window):  # Apply a simple moving average using a uniform convolution kernel.
    kernel = np.ones(window) / window  # Normalize so smoothing does not rescale amplitude.
    return np.convolve(data, kernel, mode="valid")  # 'valid' trims edges where the kernel would spill outside data.


def _align_episodes(raw: list, smooth_win: int):  # Choose window ≤ data length so x/y stay aligned after smoothing.
    arr = np.asarray(raw, dtype=float)  # Work on a float64 view for stable convolution.
    n = len(arr)  # Episode count for this run.
    if n == 0:  # Empty series should never plot — caller guards earlier.
        return np.array([]), []  # Defensive empty return.
    if n < 2:  # Too short for convolution — plot raw points on their natural episode indices.
        return arr, list(range(1, n + 1))  # Episodes 1…n without smoothing.
    win = min(smooth_win, n)  # Never request a wider window than available samples.
    if win < 2:  # Should not trigger given n ≥ 2 but keeps static analyzers calm.
        return arr, list(range(1, n + 1))  # Fall back to raw plotting.
    smoothed = smooth(arr, win)  # Length == n - win + 1.
    eps_aligned = list(range(win, n + 1))  # Episode numbers align with valid convolution tail.
    return smoothed, eps_aligned  # Paired y and x arrays for plt.plot.


def load_results(base_dir: str) -> dict:  # Read every available agent JSON under base_dir/results.
    results = {}  # Map short key → parsed dict with rewards, water, deviations lists.
    results_dir = os.path.join(base_dir, "results")  # Standard output folder created by train.py.
    for name in AGENT_NAMES:  # Walk the predetermined roster in stable order.
        path = os.path.join(results_dir, f"{name}.json")  # File name matches lowercase stem.
        if os.path.exists(path):  # Skip agents that have not been trained yet.
            with open(path, encoding="utf-8") as f:  # UTF-8 keeps Windows/macOS/Linux consistent.
                results[name] = json.load(f)  # Store entire structure including metadata fields.
        else:  # Inform the user instead of failing silently when a file is missing.
            print(f"  [warn] missing: {path}")  # Human-readable diagnostic on stderr/stdout.
    return results  # Possibly partial dict if some runs omitted.


def plot_all(base_dir: str) -> str | None:  # Render PNG to disk; returns path or None if nothing to plot.
    results = load_results(base_dir)  # Gather every JSON payload we can find.
    if not results:  # Avoid building an empty figure when training has not produced logs yet.
        print("No results found. Run train.py first.")  # Actionable guidance for the operator.
        return None  # Signal failure to callers.

    fig = plt.figure(figsize=(14, 10))  # Wide figure so the top reward panel has room for seven curves.
    fig.suptitle(  # Single title tying the three metrics together.
        "Farm RL — Agent comparison (simulated seasonal environment)",  # Descriptive heading.
        fontsize=14,  # Slightly larger than axis titles.
        fontweight="bold",  # Emphasis without additional color.
        y=0.98,  # Keep title inside the bounding box after tight_layout/savefig.
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)  # 2×2 grid with custom spacing.
    ax1 = fig.add_subplot(gs[0, :])  # Top row spans both columns → wide reward axis.
    ax2 = fig.add_subplot(gs[1, 0])  # Bottom-left water usage axis.
    ax3 = fig.add_subplot(gs[1, 1])  # Bottom-right growth deviation axis.

    for name, data in results.items():  # Plot one colored trajectory triplet per trained agent.
        color = COLORS.get(name, "#888888")  # Fall back to medium gray if a new key slips in.
        label = data["agent"]  # Prefer pretty-printed agent field stored inside JSON.
        rewards_s, eps_r = _align_episodes(data["rewards"], SMOOTH_WIN)  # Smoothed rewards with matching abscissa.
        water_s, eps_w = _align_episodes(data["water"], SMOOTH_WIN)  # Smoothed irrigation counts.
        deviations_s, eps_d = _align_episodes(data["deviations"], SMOOTH_WIN)  # Smoothed terminal deviation metric.

        ax1.plot(eps_r, rewards_s, label=label, color=color, linewidth=1.5)  # Overlay cumulative reward curve.
        ax2.plot(eps_w, water_s, label=label, color=color, linewidth=1.5)  # Overlay irrigation trajectory.
        ax3.plot(eps_d, deviations_s, label=label, color=color, linewidth=1.5)  # Overlay deviation trajectory.

    ax1.set_title("Total reward per episode (smoothed)", fontsize=11)  # Describe the vertical metric.
    ax1.set_xlabel("Episode")  # Horizontal progression through training.
    ax1.set_ylabel("Total reward")  # Sum of daily rewards within each 90-day season.
    ax1.legend(fontsize=9, loc="lower right")  # Place legend where curves usually leave headroom.
    ax1.grid(True, alpha=0.2)  # Light grid for easier reading without clutter.
    ax1.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)  # Reference line at zero reward.

    ax2.set_title("Water used per episode (pump activations)", fontsize=11)  # Count of irrigate actions.
    ax2.set_xlabel("Episode")  # Training iteration index.
    ax2.set_ylabel("Irrigation events")  # Integer pump firings per simulated year.
    ax2.legend(fontsize=9)  # Default legend placement inside axes.
    ax2.grid(True, alpha=0.2)  # Subtle grid lines.

    ax3.set_title("Final growth deviation from target curve", fontsize=11)  # End-of-episode biomass gap.
    ax3.set_xlabel("Episode")  # Same abscissa as other panels for visual comparison.
    ax3.set_ylabel("|actual − expected| growth")  # Absolute error versus agronomic schedule at day 90.
    ax3.legend(fontsize=9)  # Identify curves when printed in black and white.
    ax3.grid(True, alpha=0.2)  # Light grid for orientation.

    out_path = os.path.join(base_dir, "results", "comparison.png")  # Collocate figure beside JSON logs.
    plt.savefig(out_path, dpi=150, bbox_inches="tight")  # High enough resolution for reports.
    plt.close(fig)  # Free RAM immediately; avoids leaking figures in batch training.
    print(f"Plot saved -> {out_path}")  # Echo final destination for notebooks and CI logs.
    return out_path  # Lets train.py confirm success programmatically.


def print_summary(results: dict) -> None:  # Pretty table mirroring numeric columns from the figure legends.
    print(f"\n{'=' * 65}")  # Top rule.
    print(f"{'Agent':<25} {'Avg reward':>13} {'Water/ep':>9} {'Deviation':>10}")  # Column headers.
    print(f"{'-' * 65}")  # Subtle divider.
    for _name, data in results.items():  # Iterate arbitrary dict order — sorting not required for diagnostics.
        r = round(float(np.mean(data["rewards"][-100:])), 4)  # Tail average stabilizes noisy last episodes.
        w = round(float(np.mean(data["water"][-100:])), 1)  # Same trailing window for irrigation stats.
        d = round(float(np.mean(data["deviations"][-100:])), 4)  # Mean absolute deviation over tail.
        print(f"  {data['agent']:<23} {r:>13.4f} {w:>9.1f} {d:>10.4f}")  # Padded row for alignment.
    print(f"{'=' * 65}\n")  # Bottom rule plus breathing room.


if __name__ == "__main__":  # Allow manual regeneration from the command line.
    here = os.path.dirname(os.path.abspath(__file__))  # This file’s directory equals project root submission/.
    print("Loading results from results/ ...")  # User-facing progress string.
    res = load_results(here)  # Parse neighboring JSON snapshots.
    print_summary(res)  # Dump numeric recap to the terminal.
    plot_all(here)  # Emit the PNG alongside the data.
