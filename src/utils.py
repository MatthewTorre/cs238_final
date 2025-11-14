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


def compute_action_distribution(actions: Iterable[str]) -> Dict[str, float]:
    """Compute percentage of each action type."""
    from collections import Counter
    actions_list = list(actions)
    counts = Counter(actions_list)
    total = len(actions_list)
    return {
        "pct_habitual": counts.get("habitual", 0) / total,
        "pct_deliberate": counts.get("deliberate", 0) / total,
        "pct_rest": counts.get("rest", 0) / total,
    }


def compute_fatigue_metrics(fatigue: Iterable[float]) -> Dict[str, float]:
    """Compute fatigue statistics."""
    fatigue_arr = np.array(list(fatigue), dtype=float)
    return {
        "mean_fatigue": float(fatigue_arr.mean()),
        "max_fatigue": float(fatigue_arr.max()),
        "min_fatigue": float(fatigue_arr.min()),
        "fatigue_std": float(fatigue_arr.std()),
        "time_above_70": float((fatigue_arr > 0.7).mean()),
        "time_below_30": float((fatigue_arr < 0.3).mean()),
    }


def compute_contextual_success_rates(
    success: Iterable[float],
    difficulty: Iterable[float],
    fatigue: Iterable[float],
) -> Dict[str, float]:
    """Success rates in different contexts."""
    success_arr = np.array(list(success), dtype=float)
    difficulty_arr = np.array(list(difficulty), dtype=float)
    fatigue_arr = np.array(list(fatigue), dtype=float)

    easy_mask = difficulty_arr < 0.5
    hard_mask = difficulty_arr >= 0.5
    low_fatigue_mask = fatigue_arr < 0.4
    high_fatigue_mask = fatigue_arr >= 0.4

    return {
        "overall_success_rate": float(success_arr.mean()),
        "success_rate_easy": float(success_arr[easy_mask].mean()) if easy_mask.any() else 0.0,
        "success_rate_hard": float(success_arr[hard_mask].mean()) if hard_mask.any() else 0.0,
        "success_rate_low_fatigue": float(success_arr[low_fatigue_mask].mean()) if low_fatigue_mask.any() else 0.0,
        "success_rate_high_fatigue": float(success_arr[high_fatigue_mask].mean()) if high_fatigue_mask.any() else 0.0,
    }


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


def plot_policy_comparison(results: Dict[str, Dict], metric: str, out_path: str | Path) -> None:
    """Plot multiple policies' trajectories on same plot."""
    plt.figure(figsize=(12, 6))

    for policy_name, policy_data in results.items():
        trajectories = policy_data["all_trajectories"]
        stacked = np.array([ep[metric] for ep in trajectories], dtype=float)
        mean_traj = stacked.mean(axis=0)
        std_traj = stacked.std(axis=0)

        timesteps = range(len(mean_traj))
        plt.plot(timesteps, mean_traj, label=policy_name, linewidth=2)
        plt.fill_between(timesteps, mean_traj - std_traj, mean_traj + std_traj, alpha=0.2)

    plt.xlabel("Time Step")
    plt.ylabel(metric.capitalize())
    plt.title(f"{metric.capitalize()} Comparison Across Policies")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_summary_comparison(results: Dict[str, Dict], out_path: str | Path) -> None:
    """Bar chart comparing key metrics across policies."""
    policies = list(results.keys())
    metrics = ["total_reward_mean", "cumulative_fatigue_mean", "reward_per_effort_ratio_mean"]
    metric_labels = ["Total Reward", "Cumulative Fatigue", "Reward/Effort Ratio"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [results[p]["summary"][metric] for p in policies]
        errors = [results[p]["summary"][metric.replace("_mean", "_std")] for p in policies]

        axes[idx].bar(range(len(policies)), values, yerr=errors, capsize=5, alpha=0.7)
        axes[idx].set_xticks(range(len(policies)))
        axes[idx].set_xticklabels(policies, rotation=45, ha="right")
        axes[idx].set_title(label)
        axes[idx].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_fatigue_action_heatmap(fatigue: List[float], actions: List[str], out_path: str | Path) -> None:
    """Heatmap showing which actions are chosen at different fatigue levels."""
    fatigue_bins = 10
    action_counts = {"habitual": np.zeros(fatigue_bins), "deliberate": np.zeros(fatigue_bins), "rest": np.zeros(fatigue_bins)}

    for f, a in zip(fatigue, actions):
        bin_idx = min(int(f * fatigue_bins), fatigue_bins - 1)
        if a in action_counts:
            action_counts[a][bin_idx] += 1

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(fatigue_bins)
    colors = {"habitual": "blue", "deliberate": "red", "rest": "green"}

    for action, counts in action_counts.items():
        ax.bar(range(fatigue_bins), counts, bottom=bottom, label=action, alpha=0.8, color=colors.get(action, "gray"))
        bottom += counts

    ax.set_xlabel("Fatigue Level")
    ax.set_ylabel("Action Count")
    ax.set_xticks(range(fatigue_bins))
    ax.set_xticklabels([f"{i/fatigue_bins:.1f}" for i in range(fatigue_bins)])
    ax.legend()
    ax.set_title("Action Selection by Fatigue Level")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_state_trajectory(fatigue: List[float], difficulty: List[float], actions: List[str], out_path: str | Path) -> None:
    """2D plot showing fatigue vs difficulty trajectory with actions colored."""
    action_colors = {"habitual": "blue", "deliberate": "red", "rest": "green"}

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot trajectory lines
    for i in range(len(fatigue) - 1):
        color = action_colors.get(actions[i], "gray")
        ax.plot(
            [difficulty[i], difficulty[i + 1]],
            [fatigue[i], fatigue[i + 1]],
            color=color,
            alpha=0.1,
            linewidth=0.5,
        )

    # Plot points
    for action, color in action_colors.items():
        mask = [a == action for a in actions]
        f = np.array(fatigue)[mask]
        d = np.array(difficulty)[mask]
        if len(f) > 0:
            ax.scatter(d, f, c=color, label=action, alpha=0.6, s=20)

    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Fatigue")
    ax.set_title("State Space Trajectory")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
