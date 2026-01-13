# Project Report: Navigation with Deep Q-Learning

## Learning Algorithm

This project implements two variants of Deep Q-Networks (DQN) to solve the Banana Collector environment:

### Double DQN

Double DQN addresses the overestimation bias present in standard DQN. In standard DQN, the target network both selects and evaluates the best action, which can lead to overestimation. Double DQN fixes this by using the local network to select the action and the target network to evaluate it.

Standard DQN computes: Q_target = reward + gamma * max(Q_target(next_state))

Double DQN computes: best_action = argmax(Q_local(next_state)), then Q_target = reward + gamma * Q_target(next_state, best_action)

This decoupling reduces overestimation and leads to more stable learning.

### Dueling DQN

Dueling DQN extends the network architecture by splitting the final layers into two streams:

1. **Value Stream:** Estimates V(s), which represents how good it is to be in state s
2. **Advantage Stream:** Estimates A(s,a), which represents the advantage of taking each action in state s

These streams are combined using: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))

This architecture helps the network learn which states are valuable independent of the action taken, which is particularly useful when many actions have similar Q-values.

### Core Components

Both implementations share these components:

**Experience Replay:**
- Stores transitions (state, action, reward, next_state, done) in a replay buffer
- Random sampling breaks correlations between consecutive experiences
- Improves sample efficiency and stabilizes training

**Target Network:**
- Separate network with frozen weights updated via soft updates
- Provides stable Q-value targets during training
- Prevents the network from chasing a moving target

**Epsilon-Greedy Exploration:**
- Balances exploration (random actions) with exploitation (learned policy)
- Epsilon decays from 1.0 to 0.01 over time
- Agent gradually becomes more confident in learned behavior

## Hyperparameters

All hyperparameters were kept consistent between Double DQN and Dueling DQN implementations:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Buffer Size | 100,000 | Maximum size of experience replay buffer |
| Batch Size | 64 | Number of experiences sampled per training step |
| Gamma (γ) | 0.99 | Discount factor for future rewards |
| Tau (τ) | 0.001 | Soft update parameter for target network |
| Learning Rate | 0.0005 | Step size for gradient descent optimizer (Adam) |
| Update Every | 4 | Frequency of learning updates (every 4 timesteps) |
| Epsilon Start | 1.0 | Initial exploration rate |
| Epsilon End | 0.01 | Minimum exploration rate |
| Epsilon Decay | 0.995 | Multiplicative decay factor per episode |
| Max Episodes | 2000 | Maximum training episodes |
| Max Timesteps | 1000 | Maximum steps per episode |

These hyperparameters were not extensively tuned. They represent commonly used values from the DQN literature and worked well for this environment.

## Neural Network Architecture

### Double DQN Architecture

The network consists of three fully connected layers:

- **Input Layer:** Takes the 37-dimensional state vector
- **Hidden Layer 1:** 64 neurons with ReLU activation
- **Hidden Layer 2:** 64 neurons with ReLU activation  
- **Output Layer:** 4 neurons (one per action) with no activation function

The output layer produces raw Q-values for each action. No activation function is used because Q-values can be negative. The network has approximately 5,000 trainable parameters.

### Dueling DQN Architecture

The Dueling DQN uses the same first two layers as Double DQN, but splits after the second hidden layer:

- **Input Layer:** Takes the 37-dimensional state vector
- **Hidden Layer 1:** 64 neurons with ReLU activation
- **Hidden Layer 2:** 64 neurons with ReLU activation

After the second hidden layer, the network splits into two streams:

- **Value Stream:** Single fully connected layer mapping from 64 to 1 output (estimates V(s))
- **Advantage Stream:** Fully connected layer mapping from 64 to 4 outputs (estimates A(s,a) for each action)

The two streams are combined using the aggregation formula: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a))). This formula ensures that the advantage stream has zero mean, which makes the value and advantage streams identifiable during learning.

The network has approximately 5,100 trainable parameters, only slightly more than the standard architecture.

## Results

### Training Performance

**Double DQN:**
- Solved in 449 episodes
- Training time: ~15 minutes on RTX 5080
- Final average score: 13.0

**Dueling DQN:**
- Solved in 414 episodes
- Training time: ~15 minutes on RTX 5080
- Final average score: 13.01

### Comparison

Dueling DQN solved the environment 35 episodes faster than Double DQN (about 8% improvement). Both implementations significantly outperformed the expected baseline of ~1800 episodes.

The learning curves show similar progression in the early stages, with both variants learning basic navigation around episodes 100-200. The main difference appears in the convergence phase (episodes 300-500), where Dueling DQN reached the target score slightly faster.

### Training Progress

Double DQN reached the target score at episode 549:
```
Episode 100:   0.73
Episode 200:   4.89
Episode 300:   7.60
Episode 400:  10.57
Episode 500:  12.76
Episode 549:  13.00 (solved)
```

Dueling DQN reached the target score at episode 514:
```
Episode 100:   0.65
Episode 200:   4.30
Episode 300:   7.62
Episode 400:  10.81
Episode 500:  12.63
Episode 514:  13.01 (solved)
```

### Visual Results

Both implementations show clear learning curves with typical RL variance:

![Double DQN Training](double_dqn_training.png)

![Dueling DQN Training](dueling_dqn_training.png)

TensorBoard monitoring shows smooth convergence of the average score:

![Score Average](tensorboard_score_average.png)

The epsilon decay curve confirms proper exploration-exploitation balance:

![Epsilon Decay](tensorboard_epsilon.png)

### Testing

Both trained agents were tested over 10 episodes. Performance varies between episodes due to the stochastic nature of the environment, with scores ranging from 0 to 20+. This is expected behavior - the key metric is the average over many episodes, which both agents achieve.

## Ideas for Future Work

Several improvements could be explored to further enhance performance:

### 1. Prioritized Experience Replay (PER)
Instead of sampling experiences uniformly from the replay buffer, prioritized experience replay samples more frequently from experiences with high TD error. This focuses learning on the most surprising transitions and could lead to faster convergence, potentially reducing the number of episodes needed by 20-30%.

### 2. Multi-Step Returns (n-step DQN)
The current implementation uses 1-step TD targets. Using multi-step returns (e.g., 3-step or 5-step) would propagate rewards faster through the Q-value function. This tends to speed up learning especially in the early stages and works well when rewards are sparse.

### 3. Noisy Networks
Replace epsilon-greedy exploration with learned exploration through noisy linear layers. The network would learn when and how to explore, which often leads to more efficient exploration compared to random action selection.

### 4. Distributional RL (C51, QR-DQN)
Instead of learning just the expected return, distributional RL learns the full distribution of returns. This provides richer information about uncertainty and can improve performance in environments with multi-modal reward distributions.

### 5. Rainbow DQN
Rainbow combines all major DQN improvements: Double + Dueling + Prioritized Replay + Multi-step + Noisy Networks + Distributional. This represents the current state-of-the-art in value-based RL and would likely solve the environment in under 300 episodes.

### 6. Hyperparameter Tuning
The current hyperparameters work well but weren't extensively tuned. Systematic exploration of network architecture (layer sizes, depth), learning rate schedules, batch size, and exploration parameters could yield 5-15% improvements.

### 7. Vision-Based Learning
Training directly from pixel observations instead of the ray-based state representation would be more challenging but also more general. This would require significantly more episodes (around 10x) but demonstrates better generalization capabilities.

## Conclusion

Both Double DQN and Dueling DQN successfully solved the Banana Collector environment well above the required performance threshold. The 8% improvement from Dueling DQN validates the theoretical benefits of separating value and advantage estimation.

The fast training time (15 minutes per run on RTX 5080) and low episode count (414-449 episodes) demonstrate that modern DQN variants are highly effective for this type of discrete action space environment. Both implementations serve as strong baselines for future experiments with more advanced techniques.

## References

- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature.
- Van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double q-learning. AAAI.
- Wang, Z., et al. (2016). Dueling network architectures for deep reinforcement learning. ICML.
- Hessel, M., et al. (2018). Rainbow: Combining improvements in deep reinforcement learning. AAAI.