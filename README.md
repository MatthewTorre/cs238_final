# Bounded Resource Allocation in Sequential Decision-Making

This repository accompanies the final project proposal exploring how agents can allocate cognitive effort over time when facing partially observable tasks with fatigue. The code base is structured to let you prototype simplified POMDP environments, simulate bounded-resource agents, and analyze results through notebooks and final report artifacts.

## Getting Started

1. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Run a baseline simulation**
   ```bash
   python -m src.main --config configs/default.yaml
   ```
   Results are stored under `results/logs` as JSON snapshots that include reward trajectories and fatigue traces.
   - Switch bandit policy via `agent.policy` in the config: `epsilon_greedy` (default), `ucb`, `thompson`, or `linucb`. Policy-specific hyperparameters are also exposed in `configs/default.yaml`.
3. **Inspect results**
   - Load JSON logs into `notebooks/exploration.ipynb` for plotting fatigue vs. reward.
   - Export polished figures to `report/figures/` for inclusion in the final write-up.

## Project Layout

- `src/`: Environment, agent, and simulation logic.
- `configs/`: YAML files capturing environment, agent, and simulation hyperparameters.
- `results/`: Auto-created directories for logs and plots.
- `notebooks/`: Scratch space for exploratory analyses.
- `report/`: Notes and figures supporting the paper.

## Next Steps

- Implement improved belief updates (e.g., particle filters) in `agent.py`.
- Extend `env.py` to model richer task dynamics (multi-armed bandits, delegation delays).
- Add plotting utilities in `utils.py` for reward-effort frontiers.
- Track experiment metadata (git hash, config) in `simulate.py` for reproducibility.
