"""Environment dynamics for bounded-resource sequential decision making."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np

LOGGER = logging.getLogger(__name__)
Action = str


@dataclass
class EnvironmentParameters:
    """Hyperparameters controlling task difficulty, rewards, and fatigue."""

    success_easy_habitual: float = 0.80
    success_hard_habitual: float = 0.50
    success_easy_deliberate: float = 0.90
    success_hard_deliberate: float = 0.80
    habitual_effort_cost: float = 0.15
    deliberate_effort_cost: float = 0.40
    reward_easy_success: float = 1.0
    reward_hard_success: float = 1.4
    failure_penalty: float = -0.5
    reward_noise: float = 0.02
    habitual_fatigue_increment: float = 0.02
    deliberate_fatigue_increment: float = 0.08
    fatigue_failure_penalty: float = 0.08
    fatigue_success_relief: float = 0.10
    fatigue_recovery: float = 0.08
    fatigue_hard_task_penalty: float = 0.05
    observation_noise: float = 0.03
    difficulty_flip_chance: float = 0.30

    # Rest action parameters
    rest_recovery: float = 0.20
    rest_reward_penalty: float = -0.1

    # Homeostatic recovery
    passive_recovery_rate: float = 0.02  # Passive recovery proportional to fatigue (reduced to prevent over-recovery)

    # Fatigue-dependent performance penalties
    deliberate_fatigue_penalty: float = 0.6  # How much fatigue reduces deliberate success (0.6 = 60% reduction at max fatigue)
    habitual_fatigue_penalty: float = 0.2  # How much fatigue reduces habitual success (0.2 = 20% reduction at max fatigue)


class DecisionEnvironment:
    """Environment with binary difficulty and continuous fatigue."""

    ACTIONS = ("habitual", "deliberate", "rest")

    def __init__(self, params: EnvironmentParameters, rng: np.random.Generator | None = None):
        self.params = params
        self.rng = rng or np.random.default_rng()
        self._fatigue: float = 0.0
        self._difficulty: int = 0
        self._last_effort: float = 0.0

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #
    def reset(self) -> Dict[str, float]:
        """Reset latent state and return the initial noisy observation."""
        self._fatigue = float(self.rng.uniform(0.0, 0.2))
        self._difficulty = int(self.rng.integers(0, 2))
        LOGGER.debug("Environment reset -> fatigue=%.3f, difficulty=%s", self._fatigue, self._difficulty)
        return self.get_observation()

    def get_observation(self) -> Dict[str, float]:
        """Return the observation available to the agent."""
        fatigue_signal = np.clip(
            self._fatigue + float(self.rng.normal(0.0, self.params.observation_noise)),
            0.0,
            1.0,
        )
        return {"difficulty": float(self._difficulty), "fatigue": fatigue_signal}

    def step(self, action: Action) -> tuple[float, bool, float]:
        """
        Advance dynamics for the provided action.

        Args:
            action: either ``habitual`` or ``deliberate``.

        Returns:
            reward (float), success (bool), fatigue_{t+1} (float)
        """
        action = action.lower()
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action '{action}'. Expected one of {self.ACTIONS}.")

        success_prob = self._success_probability(action)
        success = bool(self.rng.random() < success_prob)
        effort = self._effort_cost(action)
        self._last_effort = effort

        reward = self._compute_reward(success, effort, action)
        self._fatigue = self._clip(self._fatigue + self._fatigue_delta(action, success))

        if self.rng.random() < self.params.difficulty_flip_chance:
            self._difficulty = 1 - self._difficulty

        LOGGER.debug(
            "step(action=%s) -> success=%s, reward=%.3f, fatigue=%.3f, difficulty=%s",
            action,
            success,
            reward,
            self._fatigue,
            self._difficulty,
        )
        return reward, success, self._fatigue

    # --------------------------------------------------------------------- #
    # Convenience accessors                                                 #
    # --------------------------------------------------------------------- #
    @property
    def difficulty(self) -> int:
        """Current binary difficulty indicator (0=ease, 1=hard)."""
        return self._difficulty

    @property
    def fatigue(self) -> float:
        """Current latent fatigue value."""
        return self._fatigue

    @property
    def last_effort(self) -> float:
        """Effort cost incurred by the most recent action."""
        return self._last_effort

    # --------------------------------------------------------------------- #
    # Internal helpers                                                      #
    # --------------------------------------------------------------------- #
    def _success_probability(self, action: Action) -> float:
        if action == "rest":
            return 1.0  # Rest always "succeeds"

        # Get base success probability
        if self._difficulty == 0:
            base_prob = (
                self.params.success_easy_habitual
                if action == "habitual"
                else self.params.success_easy_deliberate
            )
        else:
            base_prob = (
                self.params.success_hard_habitual
                if action == "habitual"
                else self.params.success_hard_deliberate
            )

        # Apply fatigue penalty (deliberate actions suffer more from fatigue)
        if action == "deliberate":
            # Deliberate actions require cognitive resources - heavily impaired by fatigue
            fatigue_penalty = self.params.deliberate_fatigue_penalty * self._fatigue
        else:
            # Habitual actions are more robust to fatigue
            fatigue_penalty = self.params.habitual_fatigue_penalty * self._fatigue

        return max(0.0, base_prob - fatigue_penalty)

    def _effort_cost(self, action: Action) -> float:
        if action == "rest":
            return 0.0  # No effort for resting
        return (
            self.params.habitual_effort_cost
            if action == "habitual"
            else self.params.deliberate_effort_cost
        )

    def _compute_reward(self, success: bool, effort: float, action: Action) -> float:
        if action == "rest":
            # Rest has opportunity cost but enables recovery
            return self.params.rest_reward_penalty
        base_reward = (
            self.params.reward_easy_success if self._difficulty == 0 else self.params.reward_hard_success
        )
        outcome = base_reward if success else self.params.failure_penalty
        noise = float(self.rng.normal(0.0, self.params.reward_noise))
        return outcome - effort + noise

    def _fatigue_delta(self, action: Action, success: bool) -> float:
        # Handle rest action
        if action == "rest":
            # Strong recovery from resting, plus passive recovery
            passive_recovery = -self.params.passive_recovery_rate * self._fatigue
            return -self.params.rest_recovery + passive_recovery

        # Standard action fatigue dynamics
        base = (
            self.params.habitual_fatigue_increment
            if action == "habitual"
            else self.params.deliberate_fatigue_increment
        )
        difficulty_term = self.params.fatigue_hard_task_penalty if self._difficulty == 1 else -self.params.fatigue_recovery
        outcome_term = -self.params.fatigue_success_relief if success else self.params.fatigue_failure_penalty

        # Add passive/homeostatic recovery proportional to current fatigue
        passive_recovery = -self.params.passive_recovery_rate * self._fatigue

        return base + difficulty_term + outcome_term + passive_recovery

    @staticmethod
    def _clip(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))
