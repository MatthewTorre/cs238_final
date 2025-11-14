"""Utility helpers for configuration loading, metrics, and visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import yaml


# --------------------------------------------------------------------------- #
# IO helpers                                                                  #
# --------------------------------------------------------------------------- #
def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(*dirs: str | Path) -> None:
    """Ensure a list of directories exist."""
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #
def compute_episode_metrics(rewards: Iterable[float], fatigue: Iterable[float], efforts: Iterable[float]) -> Dict[str, float]:
    """Return summary statistics for a single episode."""
    rewards_arr = np.array(list(rewards), dtype=float)
    fatigue_arr = np.array(list(fatigue), dtype=float)
    efforts_arr = np.array(list(efforts), dtype=float)

    total_reward = float(rewards_arr.sum())
    total_effort = float(np.maximum(efforts_arr.sum(), 1e-6))
    metrics = {
        "total_reward": total_reward,
        "cumulative_fatigue": float(fatigue_arr.sum()),
        "total_effort": total_effort,
        "reward_per_effort_ratio": float(total_reward / total_effort),
    }
    return metrics


def efficiency_curve(rewards: Iterable[float], efforts: Iterable[float]) -> List[float]:
    """Return reward-per-effort trajectory over time."""
    rewards_arr = np.array(list(rewards), dtype=float)
    efforts_arr = np.array(list(efforts), dtype=float)
    reward_cum = np.cumsum(rewards_arr)
    effort_cum = np.cumsum(efforts_arr)
    effort_cum = np.maximum(effort_cum, 1e-6)
    return (reward_cum / effort_cum).tolist()


# --------------------------------------------------------------------------- #
# Plotting                                                                    #
# --------------------------------------------------------------------------- #
def plot_trajectories(series: Dict[str, List[float]], title: str, out_path: str | Path | None = None) -> None:
    """Plot one or more trajectories."""
    for label, values in series.items():
        plt.plot(values, label=label)
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        plt.close()
    else:
        plt.show()
