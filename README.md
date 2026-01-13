# Navigation - Banana Collector

Train an agent using Deep Q-Network (Double DQN) to collect yellow bananas while avoiding blue bananas in a Unity ML-Agents environment.

**Trained on NVIDIA GeForce RTX 5080 - completed in under 30 minutes!**

## Project Overview

This project implements a Double DQN agent that learns to navigate a square world and collect yellow bananas (+1 reward) while avoiding blue bananas (-1 reward).

**Environment:**
- State space: 37 dimensions (velocity + ray-based perception)
- Action space: 4 discrete actions (forward, backward, left, right)
- Goal: Average score >= 13.0 over 100 consecutive episodes

**Results:**
- Solved in 449 episodes (4x faster than expected!)
- Average test score: 12.40+
- Implementation: Double DQN with Experience Replay
- Hardware: NVIDIA GeForce RTX 5080 16GB
- Training time: ~15 minutes

## Environment Setup

### Prerequisites
- Python 3.9
- Anaconda or Miniconda
- NVIDIA GPU with CUDA support (tested on RTX 5080)
- Windows, macOS, or Linux

### Installation

1. Create and activate conda environment:
```bash
conda create --name drlnd python=3.9 -y
conda activate drlnd
```

2. Install PyTorch with CUDA support:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download Unity Environment:
   - [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
   - [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
   - [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)

5. Extract the environment to the project directory:
```
p1_navigation/
├── Banana_Windows_x86_64/
│   └── Banana.exe
├── model.py
├── dqn_agent.py
├── Navigation.ipynb
└── ...
```

## Training the Agent

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `Navigation.ipynb`

3. Run cells sequentially:
   - Import packages
   - Initialize environment
   - Initialize agent
   - Define training function
   - Start training with `scores = dqn()`

Training takes approximately 20-60 minutes depending on hardware.

**Note on RTX 5080:** PyTorch may show a warning about sm_120 compatibility, but training works perfectly and achieves excellent results.

## Files

- `model.py` - Neural network architecture (Q-Network)
- `dqn_agent.py` - DQN agent implementation with Double DQN
- `Navigation.ipynb` - Training notebook
- `checkpoint.pth` - Saved model weights (created after training)
- `requirements.txt` - Python dependencies

## Algorithm

The agent uses Double DQN with the following features:
- Experience Replay (buffer size: 100,000)
- Target Network (soft updates with tau=0.001)
- Epsilon-greedy exploration (decay from 1.0 to 0.01)
- Neural Network: 37 → 64 → 64 → 4
- Learning rate: 0.0005
- Batch size: 64
- Discount factor (gamma): 0.99

## Results

Training progress:
```
Episode 100:   0.73
Episode 200:   4.89
Episode 300:   7.60
Episode 400:  10.57
Episode 500:  12.76
Episode 549:  13.00 (Solved in 449 episodes)
```

The agent successfully learned to navigate and collect bananas in under 500 episodes.

## Testing the Trained Agent

Load saved weights and watch the agent:
```python
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
# Run test episode
```

## Future Improvements

Potential enhancements:
- Prioritized Experience Replay
- Dueling DQN architecture
- Rainbow DQN (combination of improvements)
- Hyperparameter tuning
- Learning from pixels (visual input)

## License

This project is part of the Udacity Deep Reinforcement Learning Nanodegree.