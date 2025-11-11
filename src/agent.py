"""Agent definitions for bounded-resource allocation experiments."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

LOGGER = logging.getLogger(__name__)
Action = str


@dataclass
class AgentParameters:
    """Hyperparameters controlling action selection."""

    epsilon: float = 0.1
    alpha: float = 0.2
    initial_efficiency: float = 1.0
    fatigue_penalty: float = 0.5
    habitual_hard_penalty: float = 0.2
    deliberate_easy_penalty: float = 0.05
    min_effort: float = 1e-3


@dataclass
class Agent:
    """Epsilon-greedy agent that tracks reward-per-effort efficiency."""

    params: AgentParameters
    rng: np.random.Generator | None = None
    _efficiency_estimates: Dict[Action, float] = field(init=False)

    def __post_init__(self) -> None:
        self.rng = self.rng or np.random.default_rng()
        self._efficiency_estimates = {}
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Reset internal estimates before a new episode."""
        self._efficiency_estimates = {
            "habitual": self.params.initial_efficiency,
            "deliberate": self.params.initial_efficiency,
        }

    def select_action(self, fatigue: float, difficulty: float) -> Action:
        """
        Choose an action using epsilon-greedy exploration.

        The score for each action is the agent's running estimate of reward
        per unit effort corrected by the current fatigue and difficulty.
        """
        if self.rng.random() < self.params.epsilon:
            action = self.rng.choice(["habitual", "deliberate"])
            LOGGER.debug("Agent exploring -> %s", action)
            return action  # explore

        scores = {
            action: self._scored_efficiency(action, fatigue, difficulty)
            for action in ("habitual", "deliberate")
        }
        action = max(scores, key=scores.get)
        LOGGER.debug("Agent exploiting -> %s (scores=%s)", action, scores)
        return action

    def observe_outcome(self, action: Action, reward: float, effort: float) -> None:
        """Update efficiency estimates after observing environment feedback."""
        denominator = max(effort, self.params.min_effort)
        target = reward / denominator
        old = self._efficiency_estimates[action]
        self._efficiency_estimates[action] = old + self.params.alpha * (target - old)
        LOGGER.debug(
            "Agent update action=%s reward=%.3f effort=%.3f target=%.3f new_est=%.3f",
            action,
            reward,
            effort,
            target,
            self._efficiency_estimates[action],
        )

    # ------------------------------------------------------------------ #
    def _scored_efficiency(self, action: Action, fatigue: float, difficulty: float) -> float:
        """Return efficiency penalized by fatigue/difficulty context."""
        estimate = self._efficiency_estimates[action]
        fatigue_penalty = self.params.fatigue_penalty * fatigue
        difficulty_penalty = 0.0
        if action == "habitual" and difficulty >= 0.5:
            difficulty_penalty += self.params.habitual_hard_penalty
        if action == "deliberate" and difficulty < 0.5:
            difficulty_penalty += self.params.deliberate_easy_penalty
        return estimate - fatigue_penalty - difficulty_penalty
