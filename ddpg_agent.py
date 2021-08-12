import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = 10000  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 9e-4              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0001   # L2 weight decay
LEARN_PERIOD = 500
TIMES_TO_LEARN = 11
ACTION_PENALTY = 0.1
DIST_CHANGE_MODIFIER = 0.01
BAD_HEIGHT_MODIFIER = 0.05

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using this device: {device}')

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed):
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
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise_std = 0.3
        #self.noise = np.random.normal(np.zeros([4]), np.array([0.2]*4))
        #self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.time_since_last_learn = 0
        
        self.ep_count = 0
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        '''
        dist_1 = state[:,13] - state[:,26]
        dist_2 = state[:,15] - state[:,27]
        dist_3 = state[:,14]
        total_dist = np.sum(np.power(np.power(dist_1, 2) + np.power(dist_2, 2) + np.power(dist_3, 2), 0.5))
        
        dist_1 = next_state[:,13] - next_state[:,26]
        dist_2 = next_state[:,15] - next_state[:,27]
        dist_3 = next_state[:,14]
        total_next_dist = np.sum(np.power(np.power(dist_1, 2) + np.power(dist_2, 2) + np.power(dist_3, 2), 0.5))
        
        dist_change = total_next_dist - total_dist
        dist_change *= DIST_CHANGE_MODIFIER

        bad_height_penalty = np.sum(np.abs(next_state[:,14])) * BAD_HEIGHT_MODIFIER
        
        self.memory.add(state, action, reward - dist_change - bad_height_penalty, next_state, done)
        '''
        # reward is a scalar...
        # and the shape is a singleton array
        
        old_x_ball = float(state[:,26])
        old_z_ball = float(state[:,27])
        new_x_ball = float(next_state[:,26])
        new_z_ball = float(next_state[:,27])
        new_pos_ball = np.array([new_x_ball, 0, new_z_ball])
        elbow_target = np.array([new_x_ball/2, -2, new_z_ball/2])
        
        distance_traveled_ball = ((new_x_ball-old_x_ball)**2 + (new_z_ball-old_z_ball)**2) ** 0.5
        
        old_x_hand = float(state[:,13])
        old_y_hand = float(state[:,14])
        old_z_hand = float(state[:,15])
        new_x_hand = float(next_state[:,13])
        new_y_hand = float(next_state[:,14])
        new_z_hand = float(next_state[:,15])
        new_pos_hand = np.array([new_x_hand, new_y_hand, new_z_hand])
        
        radius_sq_hand = new_x_hand**2 + new_z_hand**2
        radius_hand = radius_sq_hand**0.5
        distance_traveled_hand = ((new_x_hand-old_x_hand)**2 + (new_y_hand-old_y_hand)**2 + (new_z_hand-old_z_hand)**2) ** 0.5
        
        hand_from_ball_hor_dist = ((new_x_hand-new_x_ball)**2 + (new_z_hand-new_z_ball)**2) ** 0.5
        
        old_x_elbow = float(state[:,0])
        old_y_elbow = float(state[:,1])
        old_z_elbow = float(state[:,2])
        new_x_elbow = float(next_state[:,0])
        new_y_elbow = float(next_state[:,1])
        new_z_elbow = float(next_state[:,2])
        new_pos_elbow = np.array([new_x_elbow, new_y_elbow, new_z_elbow])
        
        radius_sq_elbow = new_x_elbow**2 + new_z_elbow**2
        radius_elbow = radius_sq_elbow**0.5
        distance_traveled_elbow = ((new_x_elbow-old_x_elbow)**2 + (new_y_elbow-old_y_elbow)**2 + (new_z_elbow-old_z_elbow)**2) ** 0.5
        
        elbow_error = np.linalg.norm(new_pos_elbow - elbow_target)
        hand_error = np.linalg.norm(new_pos_hand - new_pos_elbow + elbow_target - new_pos_ball)
        
        '''
        if new_y_hand < 0 and new_y_hand > -0.2:
            reward += 0.02
            if radius_hand < 8 and radius_hand > 7.8:
                reward += 0.1
        
        if new_y_elbow > -1.5:
            reward -= 1
            
        if new_y_elbow < -2.5:
            reward -= 1
            
        if radius_elbow > 2.5:
            reward -= 1
        
        if radius_elbow < 1.5:
            reward -= 1
            
        if radius_hand - radius_elbow < 0.25:
            reward -= 1
        
        if hand_from_ball_hor_dist > 4:
            reward -= 0.75
            
        if distance_traveled_hand - distance_traveled_ball > 0.1:
            reward -= 0.75
            
        if new_y_hand < -0.5:
            reward -= 0.5
            
        if new_y_hand > 0.5:
            reward -= 0.5
        
        if radius_hand < 7:
            reward -= 0.5
        '''
        reward -= elbow_error**3
        reward -= (hand_error**3)/3
        self.memory.add(state, action, reward, next_state, done)
        self.step_in_ep += 1
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            self.time_since_last_learn += 1
            if self.time_since_last_learn > LEARN_PERIOD:
                self.time_since_last_learn = 0
                self.actor_local.train()
                self.critic_local.train()
                for i in range(TIMES_TO_LEARN):
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)
                self.actor_local.eval()
                self.critic_local.eval()

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += np.random.normal(np.zeros([4]), np.array([self.noise_std]*4))
            self.noise_std *= 0.99999
            #action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.ep_count += 1
        self.step_in_ep = 0
        # only need the below line for OUNoise
        #self.noise.reset()

    def learn(self, experiences, gamma):
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
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        # in a case where the model just flails, first loss term is usually about -4. second is usually about 0.77 (before weighting)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # 14 seems to be height. 13 and 15 are horizontal coords
        #actor_loss += torch.sum(torch.where(torch.abs(next_states[:,13])>9, torch.abs(next_states[:,13]), torch.zeros(next_states[:,13].shape, device=device)))
        #actor_loss += torch.sum(torch.where(torch.abs(next_states[:,15])>9, torch.abs(next_states[:,15]), torch.zeros(next_states[:,15].shape, device=device)))
        #actor_loss += torch.sum(torch.abs(next_states[:,14]))
        
        # + ACTION_PENALTY * torch.norm(actions_pred)
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
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state * 0.33

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