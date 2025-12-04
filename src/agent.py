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

    policy: str = "epsilon_greedy"  # epsilon_greedy | ucb | thompson | linucb
    epsilon: float = 0.1
    alpha: float = 0.2
    initial_efficiency: float = 1.0
    fatigue_penalty: float = 0.5
    habitual_hard_penalty: float = 0.2
    deliberate_easy_penalty: float = 0.05
    min_effort: float = 1e-3
    ucb_c: float = 1.0
    thompson_prior_mean: float = 0.0
    thompson_prior_var: float = 1.0
    thompson_obs_var: float = 1.0
    linucb_alpha: float = 1.0
    linucb_dim: int = 3  # bias, fatigue, difficulty


@dataclass
class Agent:
    """Agent supporting multiple bandit-style policies on reward-per-effort."""

    params: AgentParameters
    rng: np.random.Generator | None = None
    _efficiency_estimates: Dict[Action, float] = field(init=False)
    _counts: Dict[Action, int] = field(init=False)
    _means: Dict[Action, float] = field(init=False)
    _post_mean: Dict[Action, float] = field(init=False)
    _post_precision: Dict[Action, float] = field(init=False)
    _A: Dict[Action, np.ndarray] = field(init=False)
    _b: Dict[Action, np.ndarray] = field(init=False)
    _timestep: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.rng = self.rng or np.random.default_rng()
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Reset internal estimates before a new episode."""
        self._timestep = 0
        self._efficiency_estimates = {
            "habitual": self.params.initial_efficiency,
            "deliberate": self.params.initial_efficiency,
        }
        self._counts = {"habitual": 0, "deliberate": 0}
        self._means = {"habitual": 0.0, "deliberate": 0.0}
        self._post_mean = {
            "habitual": self.params.thompson_prior_mean,
            "deliberate": self.params.thompson_prior_mean,
        }
        prior_precision = 1.0 / max(self.params.thompson_prior_var, 1e-6)
        self._post_precision = {"habitual": prior_precision, "deliberate": prior_precision}
        dim = self.params.linucb_dim
        self._A = {
            "habitual": np.eye(dim, dtype=float),
            "deliberate": np.eye(dim, dtype=float),
        }
        self._b = {
            "habitual": np.zeros((dim, 1), dtype=float),
            "deliberate": np.zeros((dim, 1), dtype=float),
        }

    def select_action(self, fatigue: float, difficulty: float) -> Action:
        """
        Choose an action using epsilon-greedy exploration.

        The score for each action is the agent's running estimate of reward
        per unit effort corrected by the current fatigue and difficulty.
        """
        self._timestep += 1
        policy = self.params.policy.lower()
        if policy == "epsilon_greedy":
            if self.rng.random() < self.params.epsilon:
                action = self.rng.choice(["habitual", "deliberate"])
                LOGGER.debug("Agent exploring -> %s", action)
                return action
            scores = {a: self._scored_efficiency(a, fatigue, difficulty) for a in ("habitual", "deliberate")}
        elif policy == "ucb":
            scores = {a: self._ucb_score(a, fatigue, difficulty) for a in ("habitual", "deliberate")}
        elif policy == "thompson":
            scores = {a: self._thompson_sample(a, fatigue, difficulty) for a in ("habitual", "deliberate")}
        elif policy == "linucb":
            features = self._feature_vector(fatigue, difficulty)
            scores = {a: self._linucb_score(a, features, fatigue, difficulty) for a in ("habitual", "deliberate")}
        else:
            raise ValueError(f"Unknown policy '{self.params.policy}'.")

        action = max(scores, key=scores.get)
        LOGGER.debug("Policy=%s -> %s (scores=%s)", policy, action, scores)
        return action

    def observe_outcome(self, action: Action, reward: float, effort: float, fatigue: float, difficulty: float) -> None:
        """Update efficiency estimates after observing environment feedback."""
        target = self._target(reward, effort)
        policy = self.params.policy.lower()

        # Shared count/mean updates
        self._counts[action] += 1
        count = self._counts[action]
        prev_mean = self._means[action]
        self._means[action] = prev_mean + (target - prev_mean) / float(count)

        if policy == "epsilon_greedy":
            old = self._efficiency_estimates[action]
            self._efficiency_estimates[action] = old + self.params.alpha * (target - old)
        elif policy == "thompson":
            # Conjugate Gaussian update for reward/effort ratio
            obs_prec = 1.0 / max(self.params.thompson_obs_var, 1e-6)
            prior_prec = self._post_precision[action]
            prior_mean = self._post_mean[action]
            post_prec = prior_prec + obs_prec
            post_mean = (prior_prec * prior_mean + obs_prec * target) / post_prec
            self._post_precision[action] = post_prec
            self._post_mean[action] = post_mean
        elif policy == "linucb":
            features = self._feature_vector(fatigue, difficulty)
            x = features.reshape((-1, 1))
            A = self._A[action]
            b = self._b[action]
            self._A[action] = A + x @ x.T
            self._b[action] = b + target * x

        LOGGER.debug(
            "Agent update policy=%s action=%s reward=%.3f effort=%.3f target=%.3f mean=%.3f",
            policy,
            action,
            reward,
            effort,
            target,
            self._means[action],
        )

    # ------------------------------------------------------------------ #
    def _scored_efficiency(self, action: Action, fatigue: float, difficulty: float) -> float:
        """Return efficiency penalized by fatigue/difficulty context."""
        estimate = self._efficiency_estimates[action]
        return estimate - self._context_penalty(action, fatigue, difficulty)

    def _ucb_score(self, action: Action, fatigue: float, difficulty: float) -> float:
        count = self._counts[action]
        if count == 0:
            bonus = float("inf")
        else:
            bonus = self.params.ucb_c * np.sqrt(2.0 * np.log(max(self._timestep, 1)) / count)
        return self._means[action] + bonus - self._context_penalty(action, fatigue, difficulty)

    def _thompson_sample(self, action: Action, fatigue: float, difficulty: float) -> float:
        var = 1.0 / self._post_precision[action]
        sample = float(self.rng.normal(self._post_mean[action], np.sqrt(var)))
        return sample - self._context_penalty(action, fatigue, difficulty)

    def _linucb_score(self, action: Action, features: np.ndarray, fatigue: float, difficulty: float) -> float:
        A = self._A[action]
        b = self._b[action]
        A_inv = np.linalg.inv(A)
        theta = A_inv @ b
        x = features.reshape((-1, 1))
        mean = float((theta.T @ x)[0, 0])
        bonus = self.params.linucb_alpha * np.sqrt(float((x.T @ A_inv @ x)[0, 0]))
        return mean + bonus - self._context_penalty(action, fatigue, difficulty)

    def _context_penalty(self, action: Action, fatigue: float, difficulty: float) -> float:
        fatigue_penalty = self.params.fatigue_penalty * fatigue
        difficulty_penalty = 0.0
        if action == "habitual" and difficulty >= 0.5:
            difficulty_penalty += self.params.habitual_hard_penalty
        if action == "deliberate" and difficulty < 0.5:
            difficulty_penalty += self.params.deliberate_easy_penalty
        return fatigue_penalty + difficulty_penalty

    def _target(self, reward: float, effort: float) -> float:
        denominator = max(effort, self.params.min_effort)
        return reward / denominator

    def _feature_vector(self, fatigue: float, difficulty: float) -> np.ndarray:
        base = np.array([1.0, fatigue, difficulty], dtype=float)
        dim = self.params.linucb_dim
        if dim <= base.shape[0]:
            return base[:dim]
        padded = np.zeros(dim, dtype=float)
        padded[: base.shape[0]] = base
        return padded
