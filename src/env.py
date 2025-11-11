"""Decision environment describing bounded-resource sequential decision problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass
class EnvironmentParameters:
    """Collection of environment hyperparameters."""

    fatigue_decay: float = 0.1
    fatigue_increment: float = 0.2
    reward_scale: float = 1.0
    stochasticity: float = 0.1


class DecisionEnvironment:
    """
    Simplified POMDP-like environment where hidden fatigue affects rewards.

    The environment keeps track of observable task difficulty and latent fatigue.
    """

    def __init__(self, params: EnvironmentParameters, rng: np.random.Generator | None = None):
        self.params = params
        self.rng = rng or np.random.default_rng()
        self._fatigue = 0.0
        self._difficulty = 0.5

    def reset(self) -> Dict[str, float]:
        """Reset environment state and return initial observation."""
        self._fatigue = float(self.rng.uniform(0.0, 0.2))
        self._difficulty = float(self.rng.uniform(0.3, 0.7))
        return self._observation()

    def step(self, action: str) -> Tuple[Dict[str, float], float, Dict[str, float]]:
        """
        Advance the environment using the provided action.

        Returns:
            observation (dict), reward (float), info (dict)
        """
        action = action.lower()
        base_reward = (1.0 - self._difficulty) * self.params.reward_scale
        noise = float(self.rng.normal(0.0, self.params.stochasticity))

        # Deliberative actions reduce difficulty but cost energy.
        if action == "deliberate":
            reward = base_reward + 0.5 - self._fatigue + noise
            self._fatigue += self.params.fatigue_increment
            self._difficulty = max(0.0, self._difficulty - 0.1)
        elif action == "delegate":
            reward = base_reward + noise
            self._fatigue *= 0.9
        else:  # heuristic / default
            reward = base_reward - 0.1 * self._difficulty + noise
            self._fatigue = max(0.0, self._fatigue - self.params.fatigue_decay)
            self._difficulty = min(1.0, self._difficulty + 0.05)

        info = {"fatigue": self._fatigue, "difficulty": self._difficulty}
        return self._observation(), reward, info

    def _observation(self) -> Dict[str, float]:
        """Partial observation with noise on latent fatigue."""
        observed_fatigue = self._fatigue + float(self.rng.normal(0.0, 0.05))
        return {"difficulty": self._difficulty, "fatigue_signal": observed_fatigue}
