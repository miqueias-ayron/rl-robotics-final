"""Helpers for RELATORIO.ipynb: load saved rewards, evaluate learned policies and plot learning curves."""

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


def evaluate_policy(q_table, env, n_episodes=100):
    """Run the greedy policy of `q_table` for `n_episodes`; return per-episode timesteps and success flags."""
    timesteps, successes = [], []
    for seed in range(n_episodes):
        state, _ = env.reset(seed=seed)
        done = truncated = False
        steps = 0
        while not (done or truncated):
            state, _, done, truncated, _ = env.step(int(np.argmax(q_table[state])))
            steps += 1
        timesteps.append(steps)
        successes.append(done)
    return np.array(timesteps), np.array(successes)


def evaluation_summary(q_tables, env, n_episodes=100) -> dict:
    """Evaluate the greedy policy of every run's Q-table; return timesteps (mean, std) and success rate (mean, std across runs)."""
    timesteps, successes = zip(*(evaluate_policy(q, env, n_episodes) for q in q_tables))
    timesteps, successes = np.array(timesteps), np.array(successes)
    done_steps = timesteps[successes]
    success_rates = 100 * successes.mean(axis=1)
    return {
        "Timesteps médios": done_steps.mean() if done_steps.size else np.nan,
        "Desvio padrão dos timesteps": done_steps.std(ddof=1) if done_steps.size > 1 else np.nan,
        "Taxa de sucesso (%)": success_rates.mean(),
        "Desvio da taxa de sucesso entre runs": success_rates.std(ddof=1),
    }


def results_table(rewards: dict, q_tables: dict, envs: dict, labels: dict, last_n: int = 1000) -> pd.DataFrame:
    """One row per experiment: final training reward plus the greedy evaluation of the Q-tables of all runs."""
    rows = []
    for name, r in rewards.items():
        rows.append({
            "Configuração": labels[name],
            f"Recompensa média (últ. {last_n} ep. de treino)": r[:, -last_n:].mean(),
            **evaluation_summary(q_tables[name], envs[name]),
        })
    return pd.DataFrame(rows).set_index("Configuração").round(2)


def transfer_table(q_tables: dict, deterministic_env, stochastic_env) -> pd.DataFrame:
    """Test the deterministic Q-tables in the rainy environment and compare with the other two scenarios."""
    rows = []
    for algorithm, label in [("qlearning", "Q-Learning"), ("sarsa", "SARSA")]:
        scenarios = [
            ("treino determinístico → teste determinístico", f"{algorithm}_deterministic", deterministic_env),
            ("treino determinístico → teste com chuva", f"{algorithm}_deterministic", stochastic_env),
            ("treino com chuva → teste com chuva", f"{algorithm}_stochastic", stochastic_env),
        ]
        for scenario, name, env in scenarios:
            rows.append({"Algoritmo": label, "Cenário": scenario, **evaluation_summary(q_tables[name], env)})
    return pd.DataFrame(rows).set_index(["Algoritmo", "Cenário"]).round(2)


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
