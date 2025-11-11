"""Agent definitions for bounded resource allocation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class AgentParameters:
    """Hyperparameters controlling effort policy."""

    fatigue_threshold: float = 0.6
    deliberation_cost: float = 0.2
    heuristic_bias: float = 0.1
    delegation_penalty: float = 0.05


class Agent:
    """Simple agent with a fatigue belief that drives action selection."""

    def __init__(self, params: AgentParameters, rng: np.random.Generator | None = None):
        self.params = params
        self.rng = rng or np.random.default_rng()
        self._fatigue_belief = 0.2

    def reset(self) -> None:
        self._fatigue_belief = 0.2

    def act(self, observation: Dict[str, float]) -> str:
        """Choose an action given the observation."""
        self._update_belief(observation)
        difficulty = observation.get("difficulty", 0.5)

        if self._fatigue_belief < self.params.fatigue_threshold and difficulty > 0.5:
            return "deliberate"
        if self._fatigue_belief > 0.8:
            return "delegate"
        # Add stochastic exploration so policies can improve during learning.
        return "heuristic" if self.rng.random() > self.params.heuristic_bias else "deliberate"

    def _update_belief(self, observation: Dict[str, float]) -> None:
        """Kalman-filter style update on fatigue belief."""
        signal = observation.get("fatigue_signal", 0.0)
        self._fatigue_belief = 0.8 * self._fatigue_belief + 0.2 * signal
