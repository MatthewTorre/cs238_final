"""Entry point for running bounded resource allocation experiments."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from .agent import Agent, AgentParameters

from .env import Action, DecisionEnvironment, EnvironmentParameters
from .simulate import SimulationConfig, run_simulation
from .utils import (
    compute_episode_metrics,
    efficiency_curve,
    ensure_dirs,
    load_config,
    plot_trajectories,
    compute_action_distribution,
    compute_fatigue_metrics,
    compute_contextual_success_rates,
    plot_policy_comparison,
    plot_summary_comparison,
    plot_fatigue_action_heatmap,
    plot_state_trajectory,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded resource allocation experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config describing environment and agent settings.",
    )
    parser.add_argument("--log-dir", type=str, default="results/logs", help="Where to store simulation summaries.")    
    parser.add_argument("--plot-dir", type=str, default="results/plots", help="Where to store plots.")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Fixed Policy Definitions                                                    #
# --------------------------------------------------------------------------- #
def policy_always_habitual(fatigue: float, difficulty: float) -> Action:
    """Always choose habitual action regardless of state."""
    return "habitual"


def policy_always_deliberate(fatigue: float, difficulty: float) -> Action:
    """Always choose deliberate action regardless of state."""
    return "deliberate"


def policy_threshold_fatigue(fatigue: float, difficulty: float, threshold: float = 0.4) -> Action:
    """Choose habitual when fatigue is high, deliberate when fatigue is low."""
    return "habitual" if fatigue >= threshold else "deliberate"


def policy_adaptive_rest(fatigue: float, difficulty: float) -> Action:
    """Three-way policy: rest when exhausted, habitual when tired, deliberate when fresh."""
    if fatigue >= 0.8:
        return "rest"
    elif fatigue >= 0.4:
        return "habitual"
    else:
        return "deliberate"


def policy_difficulty_aware_rest(fatigue: float, difficulty: float) -> Action:
    """Strategic policy: consider both fatigue and difficulty for optimal action choice."""
    # Very high fatigue: must rest
    if fatigue >= 0.75:
        return "rest"

    # High fatigue: only use habitual
    if fatigue >= 0.5:
        return "habitual"

    # Medium fatigue: deliberate only on easy tasks
    if fatigue >= 0.3:
        return "deliberate" if difficulty < 0.5 else "habitual"

    # Low fatigue: always deliberate to maximize reward
    return "deliberate"


def policy_always_rest(fatigue: float, difficulty: float) -> Action:
    """Baseline: always rest (for comparison)."""
    return "rest"


# --------------------------------------------------------------------------- #
# Episode Execution                                                           #
# --------------------------------------------------------------------------- #
def run_episode(
    env: DecisionEnvironment,
    policy: Callable[[float, float], Action],
    horizon: int,
) -> tuple[Dict[str, List[float]], Dict[str, float]]:
    """Run a single episode using a fixed policy and return trajectories and metrics."""
    observation = env.reset()

    episode = {"rewards": [], "fatigue": [], "effort": [], "success": [], "actions": [], "difficulty": []}

    for _ in range(horizon):
        action = policy(observation["fatigue"], observation["difficulty"])
        reward, success, fatigue_value = env.step(action)
        effort = env.last_effort

        observation = env.get_observation()
        episode["rewards"].append(reward)
        episode["fatigue"].append(fatigue_value)
        episode["effort"].append(effort)
        episode["success"].append(1.0 if success else 0.0)
        episode["actions"].append(action)
        episode["difficulty"].append(observation["difficulty"])

    episode["efficiency"] = efficiency_curve(episode["rewards"], episode["effort"])
    metrics = compute_episode_metrics(episode["rewards"], episode["fatigue"], episode["effort"])
    return episode, metrics


def run_policy_experiments(
    env: DecisionEnvironment,
    policies: Dict[str, Callable[[float, float], Action]],
    num_episodes: int,
    horizon: int,
) -> Dict[str, Dict]:
    """Run all policies for multiple episodes and collect results."""
    results = {}

    for policy_name, policy_fn in policies.items():
        LOGGER.info("Running policy: %s", policy_name)
        trajectories: List[Dict[str, List[float]]] = []
        metrics: List[Dict[str, float]] = []

        for episode_idx in range(num_episodes):
            episode_traj, episode_metrics = run_episode(env, policy_fn, horizon)
            metrics.append(episode_metrics)
            trajectories.append(episode_traj)

            LOGGER.info(
                "  Episode %d -> reward=%.3f ratio=%.3f",
                episode_idx,
                episode_metrics["total_reward"],
                episode_metrics["reward_per_effort_ratio"],
            )

        # Compute summary statistics
        summary = {}
        for key in metrics[0]:
            values = np.array([m[key] for m in metrics], dtype=float)
            summary[f"{key}_mean"] = float(values.mean())
            summary[f"{key}_std"] = float(values.std())

        # Compute additional metrics across all episodes
        all_actions = [action for traj in trajectories for action in traj["actions"]]
        all_fatigue = [f for traj in trajectories for f in traj["fatigue"]]
        all_success = [s for traj in trajectories for s in traj["success"]]
        all_difficulty = [d for traj in trajectories for d in traj["difficulty"]]

        action_dist = compute_action_distribution(all_actions)
        fatigue_stats = compute_fatigue_metrics(all_fatigue)
        success_rates = compute_contextual_success_rates(all_success, all_difficulty, all_fatigue)

        # Merge all metrics into summary
        summary.update(action_dist)
        summary.update(fatigue_stats)
        summary.update(success_rates)

        results[policy_name] = {
            "summary": summary,
            "metrics": metrics,
            "first_episode_trajectory": trajectories[0],  # Log first episode
            "all_trajectories": trajectories,  # Keep all for plotting
        }

    return results


# --------------------------------------------------------------------------- #
# Result Persistence and Plotting                                             #
# --------------------------------------------------------------------------- #
def save_results(
    results: Dict[str, Dict],
    config: SimulationConfig,
    log_dir: str,
) -> None:
    """Save results to JSON file in notebook-compatible format."""
    timestamp = int(time.time())

    # Format for notebook compatibility
    payload = {
        "config": asdict(config),
        "policies": {}
    }

    for policy_name, policy_data in results.items():
        payload["policies"][policy_name] = {
            "summary": policy_data["summary"],
            "metrics": policy_data["metrics"],
            "trajectories": [policy_data["first_episode_trajectory"]],  # Save first episode trajectory
        }

    out_path = Path(log_dir) / f"run_{timestamp}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    LOGGER.info("Saved results -> %s", out_path)
    print(f"\nResults saved to: {out_path}")


def plot_policy_results(
    results: Dict[str, Dict],
    plot_dir: str,
) -> None:
    """Generate plots for all policies using utils.py functions."""
    plot_path = Path(plot_dir)

    for policy_name, policy_data in results.items():
        trajectories = policy_data["all_trajectories"]

        # Stack trajectories for averaging
        stacked = {
            key: np.array([ep[key] for ep in trajectories], dtype=float)
            for key in ("rewards", "fatigue", "efficiency")
        }

        mean_series = {name: arr.mean(axis=0).tolist() for name, arr in stacked.items()}

        # Plot reward trajectory
        plot_trajectories(
            {"reward": mean_series["rewards"]},
            title=f"Reward trajectory - {policy_name} (mean across episodes)",
            out_path=plot_path / f"reward_{policy_name}.png",
        )

        # Plot fatigue trajectory
        plot_trajectories(
            {"fatigue": mean_series["fatigue"]},
            title=f"Fatigue trajectory - {policy_name} (mean across episodes)",
            out_path=plot_path / f"fatigue_{policy_name}.png",
        )

        # Plot efficiency trajectory
        plot_trajectories(
            {"efficiency": mean_series["efficiency"]},
            title=f"Efficiency trajectory - {policy_name} (mean across episodes)",
            out_path=plot_path / f"efficiency_{policy_name}.png",
        )

        # Plot fatigue-action heatmap for first episode
        first_ep = trajectories[0]
        plot_fatigue_action_heatmap(
            first_ep["fatigue"],
            first_ep["actions"],
            out_path=plot_path / f"fatigue_action_heatmap_{policy_name}.png",
        )

        # Plot state trajectory for first episode
        plot_state_trajectory(
            first_ep["fatigue"],
            first_ep["difficulty"],
            first_ep["actions"],
            out_path=plot_path / f"state_trajectory_{policy_name}.png",
        )

    LOGGER.info("Individual policy plots saved to %s", plot_dir)


def plot_comparison_results(
    results: Dict[str, Dict],
    plot_dir: str,
) -> None:
    """Generate comparison plots across all policies."""
    plot_path = Path(plot_dir)

    # Comparison plots for different metrics
    plot_policy_comparison(results, "rewards", out_path=plot_path / "comparison_rewards.png")
    plot_policy_comparison(results, "fatigue", out_path=plot_path / "comparison_fatigue.png")
    plot_policy_comparison(results, "efficiency", out_path=plot_path / "comparison_efficiency.png")

    # Summary bar chart
    plot_summary_comparison(results, out_path=plot_path / "summary_comparison.png")

    LOGGER.info("Comparison plots saved to %s", plot_dir)
    print(f"All plots saved to: {plot_dir}")


def print_summary(results: Dict[str, Dict]) -> None:
    """Print summary statistics for all policies."""
    print("\n" + "="*80)
    print("POLICY COMPARISON SUMMARY")
    print("="*80)

    for policy_name, policy_data in results.items():
        summary = policy_data["summary"]
        print(f"\n{policy_name.upper()}:")
        print(f"  Mean Total Reward:        {summary['total_reward_mean']:.3f} ± {summary['total_reward_std']:.3f}")
        print(f"  Mean Total Effort:        {summary['total_effort_mean']:.3f} ± {summary['total_effort_std']:.3f}")
        print(f"  Mean Reward/Effort Ratio: {summary['reward_per_effort_ratio_mean']:.3f} ± {summary['reward_per_effort_ratio_std']:.3f}")
        print(f"  Mean Cumulative Fatigue:  {summary['cumulative_fatigue_mean']:.3f} ± {summary['cumulative_fatigue_std']:.3f}")
        print(f"  Action Distribution:")
        print(f"    Habitual:   {summary['pct_habitual']*100:.1f}%")
        print(f"    Deliberate: {summary['pct_deliberate']*100:.1f}%")
        print(f"    Rest:       {summary['pct_rest']*100:.1f}%")
        print(f"  Fatigue Stats:")
        print(f"    Mean:       {summary['mean_fatigue']:.3f}")
        print(f"    Max:        {summary['max_fatigue']:.3f}")
        print(f"    Time >7:  {summary['time_above_70']*100:.1f}%")
        print(f"    Time <0.3:  {summary['time_below_30']*100:.1f}%")
        print(f"  Success Rates:")
        print(f"    Overall:        {summary['overall_success_rate']*100:.1f}%")
        print(f"    Easy tasks:     {summary['success_rate_easy']*100:.1f}%")
        print(f"    Hard tasks:     {summary['success_rate_hard']*100:.1f}%")
        print(f"    Low fatigue:    {summary['success_rate_low_fatigue']*100:.1f}%")
        print(f"    High fatigue:   {summary['success_rate_high_fatigue']*100:.1f}%")

    print("\n" + "="*80)


# --------------------------------------------------------------------------- #
# Main Entry Point                                                            #
# --------------------------------------------------------------------------- #
def main() -> None:
    logging.basicConfig(level=logging.INFO)

    args = parse_args()
    config = load_config(args.config)
        
    ensure_dirs(Path(args.log_dir).parent, args.log_dir)
    log_dir = args.log_dir
    plot_dir = args.plot_dir if hasattr(args, 'plot_dir') and args.plot_dir else config.get("simulation", {}).get("plot_dir", "results/plots")

    ensure_dirs(log_dir, plot_dir)

    # Create environment
    env_params = EnvironmentParameters(**config.get("environment", {}))
    # agent_params = AgentParameters(**config.get("agent", {}))
    env = DecisionEnvironment(env_params)

    # Get simulation parameters
    #agent = Agent(agent_params)
    #summary = run_simulation(agent, env, sim_config)
    #print("Simulation summary:", summary)
    sim_config = SimulationConfig(**config.get("simulation", {}))
    sim_config.log_dir = log_dir
    sim_config.plot_dir = plot_dir

    # Define policies
    policies = {
        "always_habitual": policy_always_habitual,
        "always_deliberate": policy_always_deliberate,
        "threshold_fatigue": policy_threshold_fatigue,
        "adaptive_rest": policy_adaptive_rest,
        "difficulty_aware_rest": policy_difficulty_aware_rest,
        "always_rest": policy_always_rest,
    }

    # Run experiments
    results = run_policy_experiments(env, policies, sim_config.num_episodes, sim_config.horizon)

    # Save results
    save_results(results, sim_config, log_dir)

    # Generate plots
    plot_policy_results(results, plot_dir)
    plot_comparison_results(results, plot_dir)

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
