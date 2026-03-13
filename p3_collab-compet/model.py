import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model.

    Maps local observation → action for one agent.
    Smaller network than P2 (256/128 vs 400/300) because Tennis
    state space (24) is simpler than Reacher (33).
    BatchNorm on first layer stabilizes MADDPG training.
    """

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int):  local observation size per agent (24)
            action_size (int): action size per agent (2)
            seed (int):        random seed
            fc1_units (int):   nodes in first hidden layer
            fc2_units (int):   nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)  # stabilizes input scale across agents
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)  # small init for stable output

    def forward(self, state):
        """Map state → action in range [-1, 1]."""
        # handle single sample during act() call (no batch dimension)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model.

    Centralized: receives ALL agents' states and actions concatenated.
    For Tennis (2 agents):
        full_state_size  = 24 * 2 = 48
        full_action_size =  2 * 2 =  4
    Actions are injected after the first layer (standard DDPG practice).
    BatchNorm on first layer for training stability.
    """

    def __init__(self, state_size, action_size, seed, fcs1_units=256, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int):  full observation size across ALL agents (48)
            action_size (int): full action size across ALL agents (4)
            seed (int):        random seed
            fcs1_units (int):  nodes in first hidden layer
            fc2_units (int):   nodes in second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.bn1  = nn.BatchNorm1d(fcs1_units)  # normalize before action injection
        self.fc2  = nn.Linear(fcs1_units + action_size, fc2_units)  # actions injected here
        self.fc3  = nn.Linear(fc2_units, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Map (all states, all actions) → Q-value."""
        xs = F.relu(self.bn1(self.fcs1(state)))
        x  = torch.cat((xs, action), dim=1)  # inject actions after first layer
        x  = F.relu(self.fc2(x))
        return self.fc3(x)