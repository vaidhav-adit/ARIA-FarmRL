# Irrigation RL — submission package

Self-contained code to train **seven** reinforcement learning agents on a **seasonal farm simulator** (phased weather, soil moisture, tank, crop growth) and compare them with a **three-panel figure**.

## Setup

```bash
cd submission
python3 -m venv .venv
.venv/bin/pip install numpy torch matplotlib
```

## Train

```bash
.venv/bin/python train.py
```

By default this runs **500 episodes per agent**, writes metrics to `results/*.json`, and saves **`results/comparison.png`** (reward, irrigations, growth deviation).

Options:

```bash
.venv/bin/python train.py --episodes 100
.venv/bin/python train.py --seed 42
.venv/bin/python train.py --no-plot    # skip figure generation
```

## Compare only (after training)

```bash
.venv/bin/python compare.py
```

Reads `results/*.json` and refreshes `results/comparison.png`.

## Layout

| File | Role |
|------|------|
| `train.py` | Trains Q-Learning, SARSA, DQN, REINFORCE (2 variants), Actor–Critic, PPO. |
| `farm_env.py` | Environment dynamics and reward definition. |
| `growth_curve.py` | Reference growth vs day for reward and deviation metrics. |
| `compare.py` | Builds the three-panel comparison chart. |
| `agents/` | One module per algorithm implementation. |
