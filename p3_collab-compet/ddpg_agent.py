import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class OUNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(len(x))
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
            field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states      = torch.from_numpy(np.vstack([e.state      for e in experiences if e is not None])).float().to(device)
        actions     = torch.from_numpy(np.vstack([e.action     for e in experiences if e is not None])).float().to(device)
        rewards     = torch.from_numpy(np.vstack([e.reward     for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones       = torch.from_numpy(np.vstack([e.done       for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class MADDPG:
    """Coordinates both DDPG agents for the Tennis environment.

    Key design decisions based on Udacity hints (project_disc):
    1. SHARED ACTOR:
       Udacity: 'each agent used the same actor network to select actions'
       
    2. SEPARATE CRITICS (centralized training):
       Each critic sees ALL states (48) and ALL actions (4).

    3. SHARED REPLAY BUFFER:
       Udacity: 'experience was added to a shared replay buffer'

    4. EXPERIENCE MIRRORING:
       Tennis is symmetric - flip Agent 0 and Agent 1 for free extra data.
       Doubles effective buffer size without extra simulation steps.
    """

    def __init__(self, state_size, action_size, random_seed):
        self.state_size  = state_size   # 24: local observation per agent
        self.action_size = action_size  # 2: movement + jump
        self.num_agents  = 2

        # --- Shared Actor (Udacity recommendation) ---
        # One policy network for both agents (self-play).
        self.actor_local  = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # --- Separate Critics per agent (centralized) ---
        # full_state_size  = 24 * 2 = 48
        # full_action_size =  2 * 2 =  4
        self.critics_local  = [Critic(state_size * 2, action_size * 2, random_seed).to(device) for _ in range(self.num_agents)]
        self.critics_target = [Critic(state_size * 2, action_size * 2, random_seed).to(device) for _ in range(self.num_agents)]
        self.critic_optimizers = [optim.Adam(c.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY) for c in self.critics_local]

        # --- OU Noise per agent (independent exploration) ---
        self.noise = [OUNoise(action_size, random_seed) for _ in range(self.num_agents)]

        # --- Shared Replay Buffer (Udacity recommendation) ---
        self.memory = ReplayBuffer(action_size * 2, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def act(self, states, add_noise=True):
        """Select actions for both agents using the shared actor.

        Decentralized execution: each agent sees only its own 24-dim observation.
        """
        self.actor_local.eval()
        with torch.no_grad():
            actions = []
            for i, state in enumerate(states):
                state_t = torch.from_numpy(state).float().to(device)
                action  = self.actor_local(state_t).cpu().data.numpy()
                if add_noise:
                    action += self.noise[i].sample()
                actions.append(np.clip(action, -1, 1))
        self.actor_local.train()
        return np.vstack(actions)

    def reset(self):
        """Reset OU noise for all agents at episode start."""
        for n in self.noise:
            n.reset()

    def step(self, states, actions, rewards, next_states, dones):
        """Store experience and trigger learning.

        Experience Mirroring:
        Tennis is symmetric → flipping Agent 0 and Agent 1 gives valid extra data.
        Cost: zero extra simulation steps.
        """
        # store original experience
        self.memory.add(
            states.flatten(), actions.flatten(),
            rewards, next_states.flatten(), dones
        )

        # mirror: swap Agent 0 and Agent 1
        self.memory.add(
            np.flip(states,      axis=0).copy().flatten(),
            np.flip(actions,     axis=0).copy().flatten(),
            np.flip(rewards).copy(),
            np.flip(next_states, axis=0).copy().flatten(),
            np.flip(dones).copy()
        )

        # learn once buffer has enough samples
        c_loss, a_loss = 0.0, 0.0
        if len(self.memory) > BATCH_SIZE:
            for i in range(self.num_agents):
                c_loss, a_loss = self._learn(self.memory.sample(), i)
        return c_loss, a_loss

    def _learn(self, experiences, agent_idx):
        """Update critic and shared actor for one agent perspective.

        Centralized training: critic receives ALL states (48) and ALL actions (4).
        """
        states, actions, rewards, next_states, dones = experiences

        # ---- Update Critic ----

        # compute next actions for both agents using shared TARGET actor
        next_actions = torch.cat([
            self.actor_target(next_states[:, i * self.state_size:(i + 1) * self.state_size])
            for i in range(self.num_agents)
        ], dim=1)

        # TD target: r + gamma * Q_target(s', a')
        Q_targets_next = self.critics_target[agent_idx](next_states, next_actions)
        reward     = rewards[:, agent_idx].unsqueeze(1)
        done       = dones[:,   agent_idx].unsqueeze(1)
        Q_targets  = reward + (GAMMA * Q_targets_next * (1 - done))

        # minimize critic loss
        Q_expected  = self.critics_local[agent_idx](states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizers[agent_idx].zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critics_local[agent_idx].parameters(), 1)
        self.critic_optimizers[agent_idx].step()

        # ---- Update Shared Actor ----

        # recompute all actions using LOCAL actor to allow gradient flow
        all_actions = torch.cat([
            self.actor_local(states[:, i * self.state_size:(i + 1) * self.state_size])
            for i in range(self.num_agents)
        ], dim=1)

        # maximize Q-value → minimize negative Q-value
        actor_loss = -self.critics_local[agent_idx](states, all_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---- Soft Update Target Networks ----
        # theta_target = tau * theta_local + (1 - tau) * theta_target
        self._soft_update(self.critics_local[agent_idx], self.critics_target[agent_idx])
        self._soft_update(self.actor_local, self.actor_target)

        return critic_loss.item(), actor_loss.item()

    def _soft_update(self, local_model, target_model):
        """Slowly blend local weights into target network for stable learning."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)

    def save(self):
        """Save trained weights for submission."""
        torch.save(self.actor_local.state_dict(), 'checkpoint_actor.pth')
        for i, critic in enumerate(self.critics_local):
            torch.save(critic.state_dict(), f'checkpoint_critic_{i}.pth')