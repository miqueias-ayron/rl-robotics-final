"""Helpers for RELATORIO.ipynb: load saved rewards, build the summary table and plot comparisons."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MA_WINDOW = 100  # moving-average window used to make the mean curve readable
COLORS = ["#1f77b4", "#d62728"]


def load_rewards(rewards_dir: Path, name: str) -> np.ndarray:
    """Return a (n_runs, episodes) array of per-episode rewards saved by the individual notebooks."""
    return np.loadtxt(Path(rewards_dir) / f"{name}.csv", delimiter=",", ndmin=2)


def moving_average(x, window=MA_WINDOW):
    return np.convolve(x, np.ones(window) / window, mode="valid")


def first_episode_reaching(mean, threshold, window=MA_WINDOW):
    """First episode (1-indexed) where the moving average of the mean curve reaches `threshold`, or None."""
    ma = moving_average(mean, window)
    hits = np.flatnonzero(ma >= threshold)
    return int(hits[0] + window) if hits.size else None


def summary_table(rewards: dict, labels: dict, goal: float, last_n: int = 1000) -> pd.DataFrame:
    """One row per experiment with learning-speed, final-performance and stability metrics."""
    rows = []
    for name, r in rewards.items():
        mean = r.mean(axis=0)
        last = r[:, -last_n:]
        rows.append({
            "Configuração": labels[name],
            "Média 1º–1000º ep.": mean[:1000].mean(),
            f"Média últimos {last_n} ep.": last.mean(),
            f"Desvio entre runs (últ. {last_n})": last.mean(axis=1).std(ddof=1),
            f"Desvio da curva média (últ. {last_n})": mean[-last_n:].std(),
            "Ep. até MM ≥ -50": first_episode_reaching(mean, -50),
            "Ep. até MM ≥ 0": first_episode_reaching(mean, 0),
            f"% ep. com recompensa > 0 (últ. {last_n})": 100 * (last > 0).mean(),
            f"% ep. ≥ goal (últ. {last_n})": 100 * (last >= goal).mean(),
            f"% ep. ≤ -200 (≈ truncados, últ. {last_n})": 100 * (last <= -200).mean(),
        })
    return pd.DataFrame(rows).set_index("Configuração").round(2)


def plot_comparison(rewards: dict, labels: dict, names: list, title: str, goal: float, window: int = MA_WINDOW):
    """Plot the mean reward curve (raw + moving average) of each experiment in `names`, with a zoom inset."""
    n_runs, episodes = rewards[names[0]].shape
    episodes_axis = np.arange(1, episodes + 1)
    means = {name: rewards[name].mean(axis=0) for name in names}

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, color in zip(names, COLORS):
        ax.plot(episodes_axis, means[name], color=color, alpha=0.25, linewidth=0.8)
        ax.plot(episodes_axis[window - 1:], moving_average(means[name], window), color=color, linewidth=2,
                label=f"{labels[name]}: média das {n_runs} runs (MM {window})")
    ax.axhline(y=goal, color="black", linestyle="--", linewidth=1.5, label=f"Reward goal = {goal}")
    ax.set_xlabel("Episódio")
    ax.set_ylabel("Recompensa acumulada")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Zoom on the first 1000 episodes, where the curves differ the most
    inset = ax.inset_axes([0.22, 0.1, 0.36, 0.3])
    for name, color in zip(names, COLORS):
        inset.plot(episodes_axis[:1000], means[name][:1000], color=color, linewidth=1)
    inset.axhline(y=goal, color="black", linestyle="--", linewidth=1)
    inset.set_title("Primeiros 1000 episódios", fontsize=9)
    inset.tick_params(labelsize=8)
    inset.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
