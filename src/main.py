"""Entry point for running bounded resource allocation experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .agent import Agent, AgentParameters
from .env import DecisionEnvironment, EnvironmentParameters
from .simulate import SimulationConfig, run_simulation
from .utils import load_config, ensure_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded resource allocation experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config describing environment and agent settings.",
    )
    parser.add_argument("--log-dir", type=str, default="results/logs", help="Where to store simulation summaries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(Path(args.log_dir).parent, args.log_dir)

    env_params = EnvironmentParameters(**config.get("environment", {}))
    agent_params = AgentParameters(**config.get("agent", {}))
    sim_config = SimulationConfig(**config.get("simulation", {}))
    if args.log_dir:
        sim_config.log_dir = args.log_dir

    agent = Agent(agent_params)
    env = DecisionEnvironment(env_params)
    summary = run_simulation(agent, env, sim_config)
    print("Simulation summary:", summary)


if __name__ == "__main__":
    main()
