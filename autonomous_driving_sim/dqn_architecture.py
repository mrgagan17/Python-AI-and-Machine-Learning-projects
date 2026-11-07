import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# 1. Deep Q-Network (Predicting best controls from sensor arrays)
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, x):
        return self.fc(x)

# 2. DQN Agent supporting Experience Replay & epsilon-greedy updates
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95        # Discount Factor
        self.epsilon = 1.0       # Exploration Rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.lr = 0.001
        
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.update_target()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            act_values = self.model(state_t)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor(np.array([t[0] for t in minibatch]))
        actions = torch.LongTensor([t[1] for t in minibatch]).unsqueeze(1)
        rewards = torch.FloatTensor([t[2] for t in minibatch])
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch]))
        dones = torch.FloatTensor([float(t[4]) for t in minibatch])
        
        # Current Q-values
        curr_q = self.model(states).gather(1, actions).squeeze(1)
        
        # Max next Q-values
        next_q = self.target_model(next_states).max(1)[0].detach()
        
        # Calculate Targets
        targets = rewards + (1 - dones) * self.gamma * next_q
        
        loss = self.criterion(curr_q, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay