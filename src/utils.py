"""Utility helpers for configuration loading and visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import yaml
import matplotlib.pyplot as plt


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(*dirs: str) -> None:
    """Ensure a list of directories exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def plot_curves(series: Dict[str, list], out_path: str | None = None) -> None:
    """Quick plotting helper for notebooks or CLI debugging."""
    for label, values in series.items():
        plt.plot(values, label=label)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    else:
        plt.show()
