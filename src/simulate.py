"""Simulation helpers for running experiments and logging metrics."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple
import json
import time

import numpy as np

from .agent import Agent
from .env import DecisionEnvironment


@dataclass
class SimulationConfig:
    """Parameters that govern a simulation run."""

    horizon: int = 200
    num_episodes: int = 10
    log_dir: str = "results/logs"


def run_episode(agent: Agent, env: DecisionEnvironment, horizon: int) -> Dict[str, List[float]]:
    """Run a single episode and return collected trajectories."""
    obs = env.reset()
    agent.reset()
    rewards, fatigue_series = [], []

    for _ in range(horizon):
        action = agent.act(obs)
        obs, reward, info = env.step(action)
        rewards.append(reward)
        fatigue_series.append(info["fatigue"])
    return {"rewards": rewards, "fatigue": fatigue_series}


def run_simulation(agent: Agent, env: DecisionEnvironment, config: SimulationConfig) -> Dict[str, np.ndarray]:
    """Run multiple episodes and aggregate statistics."""
    episode_stats: List[Dict[str, List[float]]] = []
    for _ in range(config.num_episodes):
        episode_stats.append(run_episode(agent, env, config.horizon))

    rewards = np.array([np.sum(ep["rewards"]) for ep in episode_stats])
    fatigue = np.array([np.mean(ep["fatigue"]) for ep in episode_stats])
    summary = {"reward_mean": rewards.mean(), "reward_std": rewards.std(), "fatigue_mean": fatigue.mean()}

    _log_results(config.log_dir, episode_stats, summary)
    return summary


def _log_results(log_dir: str, trajectories: List[Dict[str, List[float]]], summary: Dict[str, float]) -> None:
    """Persist results so notebooks can reload them later."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    out_path = Path(log_dir) / f"run_{timestamp}.json"
    payload = {"summary": summary, "trajectories": trajectories}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
