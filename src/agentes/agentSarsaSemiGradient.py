import numpy as np
import gymnasium as gym
from agentes import Agent  # Importamos la clase base Agent desde su módulo

class SemiGradientSarsaAgent(Agent):
    """
    Implementación de un agente SARSA con gradiente semilineal.
    Utiliza Tile Coding para representar estados y actualizar los pesos de manera eficiente.
    """
    def __init__(self, tcenv, alpha: float, epsilon: float, decay: bool):
        self.tcenv = tcenv  # Se pasa el entorno con Tile Coding ya configurado
        self.action_space = tcenv.env.action_space.n  # Espacio de acciones del entorno original
        self.alpha = alpha  # Ajuste de alpha por tilings
        self.epsilon = epsilon
        self.decay = decay
        self.gamma = 0.99  # Factor de descuento

        # Número total de características en el aproximador lineal
        self.total_features = self.tcenv.n_tilings * np.prod(self.tcenv.bins)
        self.num_actions = self.action_space

        # Inicializamos los pesos con ceros [n_features, n_actions]
        self.w = np.zeros((self.total_features, self.num_actions))

    def q_value(self, active_features, action):
        """
        Calcula Q(s, a) como la suma de los pesos para los índices activos.
        """
        return self.w[active_features, action].sum()

    def get_action(self, active_features, n):
        """
        Selecciona una acción usando una política ε-soft adaptada a SARSA con gradiente semilineal.
        """
        if self.decay:
            self.epsilon = max(0.01, self.epsilon * 0.995)  # Decaimiento suave

        q_values = np.array([self.q_value(active_features, a) for a in range(self.num_actions)])

        best_action = np.argmax(q_values)  # Selección de la mejor acción

        action_probabilities = np.ones(self.num_actions) * (self.epsilon / self.num_actions)
        action_probabilities[best_action] += (1 - self.epsilon)

        return np.random.choice(range(self.num_actions), p=action_probabilities)

    def update(self, active_features, action, reward, next_active_feature, next_action, done):
        """
        Actualiza los pesos usando SARSA semigradiente.
        """
        q_sa = self.q_value(active_features, action)

        if done:
            td_target = reward
        else:
            q_sap = self.q_value(next_active_feature, next_action)
            td_target = reward + self.gamma * q_sap

        td_error = td_target - q_sa  # TD error

        # Actualización de los pesos con normalización por tilings
        self.w[active_features, action] += self.alpha  * td_error

