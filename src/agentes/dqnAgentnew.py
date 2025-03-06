import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from agentes import Agent  # Importamos la clase base Agent desde su módulo


# --- MODELO DQN ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, lr=0.001):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.fc(x)

# --- AGENTE DQN ---
class DQNAgentNew(Agent):
    def __init__(self, env, seed, device, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.999, lr=0.001):
        super().__init__(env)
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=50000)  # Aumentamos el tamaño de la memoria
        self.device = device
        self.feature_dim = env.tile_size * env.n_tilings
        self.num_actions = env.action_space.n
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.model = DQN(self.feature_dim, self.num_actions, lr).to(self.device)
        self.target_model = DQN(self.feature_dim, self.num_actions, lr).to(self.device)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states, targets = [], []

        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)

            with torch.no_grad():
                target = reward
                if not done:
                    target += self.gamma * torch.max(self.target_model(next_state)).item()

            q_values = self.model(state)
            target_f = q_values.clone()
            target_f[action] = target

            states.append(state)
            targets.append(target_f)

        states = torch.stack(states)
        targets = torch.stack(targets)
        
        loss = self.model.criterion(self.model(states), targets)
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
