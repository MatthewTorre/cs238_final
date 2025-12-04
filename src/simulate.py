"""Simulation helpers for running experiments and logging metrics."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .agent import Agent
from .env import DecisionEnvironment
from .utils import compute_episode_metrics, efficiency_curve, ensure_dirs, plot_trajectories

LOGGER = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Parameters that govern a simulation run."""

    horizon: int = 200
    num_episodes: int = 10
    log_dir: str = "results/logs"
    plot_dir: str = "results/plots"
    record_trajectories: bool = True


def run_episode(agent: Agent, env: DecisionEnvironment, horizon: int) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    """Run a single episode and return both trajectories and metrics."""
    observation = env.reset()
    agent.reset()

    episode = {"rewards": [], "fatigue": [], "effort": [], "success": [], "actions": []}

    for _ in range(horizon):
        action = agent.select_action(observation["fatigue"], observation["difficulty"])
        reward, success, fatigue_value = env.step(action)
        effort = env.last_effort
        agent.observe_outcome(action, reward, effort, observation["fatigue"], observation["difficulty"])

        observation = env.get_observation()
        episode["rewards"].append(reward)
        episode["fatigue"].append(fatigue_value)
        episode["effort"].append(effort)
        episode["success"].append(1.0 if success else 0.0)
        episode["actions"].append(action)

    episode["efficiency"] = efficiency_curve(episode["rewards"], episode["effort"])
    metrics = compute_episode_metrics(episode["rewards"], episode["fatigue"], episode["effort"])
    return episode, metrics


def run_simulation(agent: Agent, env: DecisionEnvironment, config: SimulationConfig) -> Dict[str, float]:
    """Run multiple episodes, aggregate statistics, persist logs, and make plots."""
    ensure_dirs(config.log_dir, config.plot_dir)

    trajectories: List[Dict[str, List[float]]] = []
    metrics: List[Dict[str, float]] = []

    for episode_idx in range(config.num_episodes):
        episode_traj, episode_metrics = run_episode(agent, env, config.horizon)
        metrics.append(episode_metrics)
        if config.record_trajectories:
            trajectories.append(episode_traj)
        LOGGER.info(
            "Episode %d -> reward=%.3f ratio=%.3f",
            episode_idx,
            episode_metrics["total_reward"],
            episode_metrics["reward_per_effort_ratio"],
        )

    summary = _summarize_metrics(metrics)
    _persist_results(trajectories, metrics, summary, config)
    if trajectories:
        _render_plots(trajectories, config)
    return summary


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _summarize_metrics(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    values = {key: np.array([m[key] for m in metrics], dtype=float) for key in metrics[0]}
    summary = {}
    for key, arr in values.items():
        summary[f"{key}_mean"] = float(arr.mean())
        summary[f"{key}_std"] = float(arr.std())
    return summary


def _persist_results(
    trajectories: List[Dict[str, List[float]]],
    metrics: List[Dict[str, float]],
    summary: Dict[str, float],
    config: SimulationConfig,
) -> None:
    timestamp = int(time.time())
    payload = {
        "config": asdict(config),
        "summary": summary,
        "metrics": metrics,
        "trajectories": trajectories,
    }
    out_path = Path(config.log_dir) / f"run_{timestamp}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    LOGGER.info("Saved run summary -> %s", out_path)


def _render_plots(trajectories: List[Dict[str, List[float]]], config: SimulationConfig) -> None:
    stacked = {key: np.array([ep[key] for ep in trajectories], dtype=float) for key in ("rewards", "fatigue", "efficiency")}

    mean_series = {name: arr.mean(axis=0).tolist() for name, arr in stacked.items()}
    plot_trajectories(
        {"reward": mean_series["rewards"]},
        title="Reward trajectory (mean across episodes)",
        out_path=Path(config.plot_dir) / "reward.png",
    )
    plot_trajectories(
        {"fatigue": mean_series["fatigue"]},
        title="Fatigue trajectory (mean across episodes)",
        out_path=Path(config.plot_dir) / "fatigue.png",
    )
    plot_trajectories(
        {"efficiency": mean_series["efficiency"]},
        title="Efficiency trajectory (mean across episodes)",
        out_path=Path(config.plot_dir) / "efficiency.png",
    )
