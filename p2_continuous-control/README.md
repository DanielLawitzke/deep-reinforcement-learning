# Continuous Control with DDPG

![Training Results](training_plot.png)

Training a robotic arm to reach and maintain target positions using deep reinforcement learning.

## Project Goal

Train a DDPG agent to control a double-jointed robotic arm in Unity's Reacher environment. The arm should keep its end effector within a moving target zone as long as possible.

Environment: Unity ML-Agents Reacher (Version 2 with 20 parallel agents)  
Target: Average score of 30.0 or higher over 100 consecutive episodes  
Result: Solved in 216 episodes (final score 30.01)

---

## Results

| Metric | Value |
|--------|-------|
| Episodes | 216 |
| Final Score | 30.01 |
| Training Time | ~1 hour |
| Hardware | NVIDIA RTX 5080 |

### Training Curves

![Moving Average](score_average.png)

100-episode moving average crossed target at episode 216

![Episode Scores](score_episode.png)

Individual episode scores showing steady improvement

![Noise Schedule](noise_scale.png)

Noise decay enabling exploration-to-exploitation transition

---

## Environment Details

### State Space
- **Dimension**: 33 continuous variables
- **Contains**: Position, rotation, velocity, and angular velocity of the arm

### Action Space
- **Dimension**: 4 continuous actions (range [-1, 1])
- **Controls**: Torque applied to the two joints

### Rewards
- **+0.1** for each timestep the end effector is in the target zone
- **Goal**: Maximize cumulative reward by staying in target

### Success Criteria
- Average score ≥30 over 100 consecutive episodes
- With 20 agents: Average across all agents per episode, then average over 100 episodes

---

## Getting Started

### Prerequisites

```bash
# Create conda environment
conda create -n drlnd python=3.8
conda activate drlnd

# Install dependencies
pip install unityagents==0.4.0
pip install torch torchvision
pip install matplotlib tensorboard
```

### Download Unity Environment

Download the appropriate version for your operating system:
- [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)

Extract to: `p2_continuous-control/Reacher_Windows_x86_64/` (or your OS equivalent)

### Training

```bash
# Start Jupyter
jupyter notebook

# Open Continuous_Control.ipynb
# Run all cells to train the agent
```

Training takes approximately 1 hour on RTX 5080.

### Using Trained Weights

```python
from ddpg_agent import Agent

# Load trained agent
agent = Agent(state_size=33, action_size=4, random_seed=0)
agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))

# Use agent for inference
actions = agent.act(states, add_noise=False)
```

---

## Algorithm: DDPG

**Deep Deterministic Policy Gradient** - Actor-Critic algorithm for continuous control

### Core Components

- **Actor Network**: μ(s|θ) → Learns deterministic policy  
- **Critic Network**: Q(s,a|w) → Evaluates state-action pairs  
- **Experience Replay**: 1M buffer for breaking temporal correlations  
- **Target Networks**: Soft updates (τ=0.001) for stable training  
- **OU Noise**: Temporally correlated exploration with decay schedule

### Network Architecture

**Actor**: `State(33) → FC(400) → ReLU → FC(300) → ReLU → Action(4) → tanh`  
**Critic**: `[State(33) → FC(400) → ReLU] + [Action(4)] → FC(300) → ReLU → Q(1)`

### Key Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| BUFFER_SIZE | 1,000,000 | Large experience replay |
| BATCH_SIZE | 256 | GPU-efficient batch size |
| LR_ACTOR | 1e-4 | Actor learning rate |
| LR_CRITIC | 1e-3 | Critic learning rate |
| noise_decay | 0.99 | Exploration schedule |
| learn_every | 20 | Update frequency |
| learn_times | 15 | Updates per step |

See `Report.md` for complete details.

---

## Critical Bug Fixes

Two implementation bugs prevented convergence and were fixed:

### Bug 1: OU Noise Implementation
**Problem**: Used `random.random()` (0-1) instead of centered Gaussian  
**Fix**: Changed to `np.random.standard_normal()` for zero-mean noise  
**Impact**: Without fix, score plateaued at 0.8-2.0

### Bug 2: Noise Application
**Problem**: Double noise + multiplication instead of addition  
**Fix**: Manual noise control with proper decay schedule  
**Impact**: Enabled smooth exploration-to-exploitation transition

**Result**: With fixes applied, training succeeded with exponential growth!

See `Report.md` for detailed technical analysis.

---

## Repository Structure

```
p2_continuous-control/
├── Continuous_Control.ipynb    # Main training notebook
├── ddpg_agent.py              # DDPG agent (with bug fixes)
├── model.py                   # Actor & Critic networks
├── checkpoint_actor.pth       # Trained actor weights
├── checkpoint_critic.pth      # Trained critic weights
├── Report.md                  # Detailed technical report
├── README.md                  # This file
└── Reacher_Windows_x86_64/    # Unity environment
```

---

## Performance Analysis

### Training Progression

| Phase | Episodes | Score | Characteristic |
|-------|----------|-------|----------------|
| 1 | 0-50 | 0→3.44 | Rapid initial learning |
| 2 | 50-100 | 3.44→9.37 | Accelerated improvement |
| 3 | 100-150 | 9.37→20.23 | Exponential growth |
| 4 | 150-216 | 20.23→30.01 | Final convergence |

### Key Observations

- Smooth exponential curve, no catastrophic forgetting
- Stable convergence, maintained score after reaching target
- Solved in 216 episodes
- No plateau failures unlike attempts without bug fixes

---

## Future Improvements

### Short Term
- **TD3**: Twin critics + delayed updates for stability
- **PER**: Prioritized experience replay for faster learning
- **Larger networks**: 4-5 layers for better capacity

### Long Term
- **SAC**: Soft Actor-Critic for better exploration
- **PPO**: Alternative on-policy approach
- **Parallel environments**: 4-10x training speedup
- **Transfer learning**: Multi-task or curriculum learning

See `Report.md` for detailed discussion of future work.

---

## Key Learnings

1. **Implementation details are critical** - Two small bugs completely prevented convergence
2. **Debugging RL is challenging** - Required systematic testing and analysis
3. **Noise scheduling matters** - Balance between exploration and exploitation is key
4. **DDPG works well when correct** - Clean, efficient algorithm for continuous control

---

## References

- [DDPG Paper (Lillicrap et al. 2016)](https://arxiv.org/abs/1509.02971)
- [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents)
- [Udacity Deep RL Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)
- [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)

---

## Acknowledgments

- **Udacity** for the Deep RL Nanodegree program
- **Unity Technologies** for ML-Agents toolkit
- **OpenAI** for foundational RL research
- **Claude (Anthropic)** for debugging assistance

---

## License

This project is part of Udacity's Deep Reinforcement Learning Nanodegree.

---

Deep Reinforcement Learning Nanodegree - Januar 2026
