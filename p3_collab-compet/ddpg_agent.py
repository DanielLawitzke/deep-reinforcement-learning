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

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, full_state_size, full_action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(full_state_size, full_action_size, random_seed).to(device)
        self.critic_target = Critic(full_state_size, full_action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, all_next_actions, all_actions):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, all_next_actions)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)  #UPDATE!
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, all_actions).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(len(x))
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class MADDPG:
    """Coordinates both DDPG agents for the Tennis environment.

    Key design decisions based on Udacity hints (project_disc):
    1. SHARED ACTOR:
       Udacity: 'each agent used the same actor network to select actions'
       → Both agents share one actor. Doubles training examples per update.
       → Hope faster convergence than two separate actors.

    2. SEPARATE CRITICS (centralized training):
       Each agent has its own critic, but it sees ALL states and ALL actions.
       → Solves non-stationarity: from Agent 0's view, Agent 1 is also learning,
         making the environment appear non-stationary. Centralized critic
         stabilizes training by having full information during training.
       → At execution time, each agent acts on local observation only.

    3. SHARED REPLAY BUFFER:
       Udacity: 'experience was added to a shared replay buffer'
       → Both agents' experiences go into one buffer.

    4. EXPERIENCE MIRRORING:
       Tennis is symmetric - both sides of the net are identical.
       → Every experience can be flipped (Agent 0 ↔ Agent 1) for free.
       → Doubles effective buffer size without extra simulation steps.
    """

    def __init__(self, state_size, action_size, random_seed):
        self.state_size = state_size    # 24: local observation per agent
        self.action_size = action_size  # 2: movement + jump
        self.num_agents = 2

        # --- Shared Actor (Udacity recommendation) ---
        # One policy network used by both agents (self-play).
        # state_size=24: each agent only sees its own local observation.
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # --- Separate Critics per agent (centralized) ---
        # full_state_size = 24 * 2 = 48: concatenated observations of both agents
        # full_action_size = 2  * 2 = 4:  concatenated actions of both agents
        self.critics_local = [
            Critic(state_size * 2, action_size * 2, random_seed).to(device)
            for _ in range(self.num_agents)
        ]
        self.critics_target = [
            Critic(state_size * 2, action_size * 2, random_seed).to(device)
            for _ in range(self.num_agents)
        ]
        self.critic_optimizers = [
            optim.Adam(c.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
            for c in self.critics_local
        ]

        # --- OU Noise per agent (independent exploration) ---
        self.noise = [OUNoise(action_size, random_seed) for _ in range(self.num_agents)]

        # --- Shared Replay Buffer (Udacity recommendation) ---
        # action_size*2 because buffer stores actions of BOTH agents concatenated
        self.memory = ReplayBuffer(action_size * 2, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def act(self, states, add_noise=True):
        """Select actions for both agents using the SHARED actor.

        Decentralized execution: each agent sees only its own 24-dim observation.
        Udacity: 'each agent used the same actor network to select actions'
        """
        self.actor_local.eval()
        with torch.no_grad():
            actions = []
            for i, state in enumerate(states):
                state_t = torch.from_numpy(state).float().to(device)
                action = self.actor_local(state_t).cpu().data.numpy()
                if add_noise:
                    action += self.noise[i].sample()
                actions.append(np.clip(action, -1, 1))
        self.actor_local.train()
        return np.vstack(actions)

    def reset(self):
        """Reset OU noise for all agents at the start of each episode."""
        for n in self.noise:
            n.reset()

    def step(self, states, actions, rewards, next_states, dones):
        """Store experience and trigger learning.

        Experience Mirroring:
        Tennis is symmetric - Agent 0 and Agent 1 play identical roles.
        Flipping Agent 0 & Agent 1 
        """
        # Store original experience
        self.memory.add(
            states.flatten(),
            actions.flatten(),
            rewards,
            next_states.flatten(),
            dones
        )

        # Mirror: swap Agent 0 and Agent 1 (symmetric environment)
        self.memory.add(
            np.flip(states, axis=0).copy().flatten(),
            np.flip(actions, axis=0).copy().flatten(),
            np.flip(rewards).copy(),
            np.flip(next_states, axis=0).copy().flatten(),
            np.flip(dones).copy()
        )

        # Learn once buffer has enough samples
        if len(self.memory) > BATCH_SIZE:
            for i in range(self.num_agents):
                self._learn(self.memory.sample(), i)

    def _learn(self, experiences, agent_idx):
        """Update critic and shared actor for one agent perspective.

        Centralized training:
        The critic receives ALL states (48-dim) and ALL actions (4-dim).
        """
        states, actions, rewards, next_states, dones = experiences

        # ---- Update Critic ----

        # Compute next actions for both agents using shared TARGET actor.
        # Split the 48-dim next_states tensor into two 24-dim observations.
        next_actions = torch.cat([
            self.actor_target(
                next_states[:, i * self.state_size:(i + 1) * self.state_size]
            )
            for i in range(self.num_agents)
        ], dim=1)

        # Compute TD target: r + gamma * Q_target(s', a')
        Q_targets_next = self.critics_target[agent_idx](next_states, next_actions)
        reward = rewards[:, agent_idx].unsqueeze(1)
        done = dones[:, agent_idx].unsqueeze(1)
        Q_targets = reward + (GAMMA * Q_targets_next * (1 - done))

        # Minimize critic loss (MSE between expected and target Q-values)
        Q_expected = self.critics_local[agent_idx](states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizers[agent_idx].zero_grad()
        critic_loss.backward()
        # Gradient clipping prevents exploding gradients (from P2 bug fix)
        torch.nn.utils.clip_grad_norm_(self.critics_local[agent_idx].parameters(), 1)
        self.critic_optimizers[agent_idx].step()

        # ---- Update Shared Actor ----

        # Recompute all actions using LOCAL actor to allow gradient flow.
        # Split states into per-agent observations for the shared actor.
        all_actions = torch.cat([
            self.actor_local(
                states[:, i * self.state_size:(i + 1) * self.state_size]
            )
            for i in range(self.num_agents)
        ], dim=1)

        # Maximize Q-value → minimize negative Q-value (gradient ascent)
        actor_loss = -self.critics_local[agent_idx](states, all_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---- Soft Update Target Networks ----
        # θ_target = τ*θ_local + (1-τ)*θ_target
        # Small τ (0.001) keeps targets stable during learning
        self._soft_update(self.critics_local[agent_idx], self.critics_target[agent_idx])
        self._soft_update(self.actor_local, self.actor_target)

    def _soft_update(self, local_model, target_model):
        """Slowly blend local weights into target network for stable learning."""
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                TAU * local_param.data + (1.0 - TAU) * target_param.data
            )

    def save(self):
        """Save trained weights for submission."""
        torch.save(self.actor_local.state_dict(), 'checkpoint_actor.pth')
        for i, critic in enumerate(self.critics_local):
            torch.save(critic.state_dict(), f'checkpoint_critic_{i}.pth')