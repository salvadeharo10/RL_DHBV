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
class DQNAgent(Agent):
    def __init__(self, env, state_dim, action_dim, device, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995, lr=0.001):
        super().__init__(env)  # Pasamos env al constructor de la clase base
        self.env = env  # Guardamos el entorno si es necesario
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=10000)
        self.device = device
                     
        # Modelos en el dispositivo correcto
        self.model = DQN(state_dim, action_dim, lr).to(self.device)
        self.target_model = DQN(state_dim, action_dim, lr).to(self.device)
        self.update_target_model()


    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Agregar dimensión batch
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)

            with torch.no_grad():
                target = reward
                if not done:
                    target += self.gamma * torch.max(self.target_model(next_state)).item()

            q_values = self.model(state)
            target_f = q_values.clone().detach()
            target_f[action] = target

            loss = self.model.criterion(q_values, target_f)
            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
