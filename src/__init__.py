"""
Bounded resource allocation package.

This module exposes convenience imports so downstream scripts can do:
```
from src import Agent, Environment
```
when running experiments.
"""

from .agent import Agent
from .env import DecisionEnvironment
from .simulate import run_simulation

__all__ = ["Agent", "DecisionEnvironment", "run_simulation"]
