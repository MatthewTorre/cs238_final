# Report Notes

## 1. What We've Built

### Environment ([src/env.py](../src/env.py))
- **State**: Binary difficulty (0/1, flips with 30% chance), continuous fatigue [0,1]
- **Actions**: Habitual (low effort/reward), Deliberate (high effort/reward), Rest (recovery)
- **Rewards**: Easy=1.0, Hard=1.4; success-dependent with -0.6 failure penalty
- **Key innovation**: Fatigue-dependent success penalties
  - Deliberate: P(success) = base - 0.6×fatigue (fails when exhausted)
  - Habitual: P(success) = base - 0.2×fatigue (robust to fatigue)
- **Fatigue dynamics**: Action accumulation + context modulation + homeostatic recovery (0.02×fatigue)

### Policies ([src/main.py](../src/main.py))
1. Always Habitual/Deliberate/Rest (baselines)
2. Threshold Fatigue (switch at 0.4)
3. Adaptive Rest (3-tier: rest@0.8, habitual@0.4, deliberate<0.4)
4. Difficulty-Aware Rest (considers both fatigue and difficulty)

### Metrics ([src/utils.py](../src/utils.py))
- Episode: total reward, effort, efficiency trajectory
- Action distribution, fatigue stats (mean, max, time in zones)
- Contextual success rates (by difficulty/fatigue)
- Visualizations: trajectories, comparisons, heatmaps, state-space plots

---

## 2. Future Steps & Development Roadmap

### Problem Framing: Restless Contextual Bandit with Hidden State

Our problem structure:
- **Arms**: {habitual, deliberate, rest}
- **Context**: (difficulty, observed_fatigue) [observable]
- **Hidden state**: true_fatigue [latent, partially observable]
- **Restless**: Fatigue evolves regardless of action
- **Non-stationary**: Reward distributions shift with fatigue state

This maps to both **POMDP** (belief-state planning) and **MAB** (online learning) frameworks.

### Two Complementary Algorithmic Paths

**Path 1: POMDP Solvers** (optimal planning with beliefs)
- Enable observation noise, implement belief tracking (particle filter)
- Solve with discrete-state value iteration or online planning (POMCP)
- Analyze value of maintaining accurate beliefs about fatigue

**Path 2: Multi-Armed Bandits** (online learning from experience)
- Classic: ε-greedy, UCB (discounted), Thompson Sampling
- Contextual: LinUCB, Neural bandits
- Structure learning: HMM, transition models, fitted Q-iteration
- Meta-learning: Transfer across difficulty distributions
- Hybrid: Bandits + rollouts, options framework

### Research Questions
1. **Learning**: Can bandits discover rest value? Adaptation speed?
2. **Context**: Does difficulty+fatigue context improve performance?
3. **Model-based vs model-free**: Sample efficiency? Asymptotic performance?
4. **Exploration**: Optimal ε? Long-term strategy discovery?
5. **Interpretability**: Extract threshold rules? Match human strategies?
6. **Non-stationarity**: Impact on regret? Adaptation speed?

---

Use this document as a living lab notebook before migrating polished text to the LaTeX report.
