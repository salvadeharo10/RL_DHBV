import numpy as np
import gymnasium as gym
from typing import List, Tuple
from agentes import Agent  # Importamos la clase base Agent desde su módulo

# Clase Monte Carlo Off-Policy Agent
class MonteCarloOffPolicyAgent(Agent):
    """
    Implementación de un agente de Aprendizaje por Refuerzo basado en Monte Carlo Off-Policy.
    Aprende una política óptima utilizando Importance Sampling para actualizar la función de valor Q(s, a).
    """
    def __init__(self, env: gym.Env, gamma: float = 0.99, epsilon: float = 0.1, decay: bool = False):
        """
        Inicializa el agente con los parámetros de entrenamiento.
        
        Parámetros:
        - env: entorno de Gym donde se entrena el agente.
        - gamma: factor de descuento que pondera la importancia de las recompensas futuras.
        - epsilon: probabilidad de exploración en la política ε-soft.
        - decay: si es True, el valor de ε disminuye a medida que avanza el entrenamiento.
        """
        super().__init__(env)
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        
        # Inicialización de las tablas Q y C (para Importance Sampling)
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        self.C = np.zeros((env.observation_space.n, env.action_space.n))

    def get_action(self, state: int, n: int) -> int:
        """
        Selecciona una acción siguiendo una política ε-soft.
        
        Parámetros:
        - state: estado actual del agente.
        - n: número de episodios transcurridos (para el decaimiento de ε si está activado).
        
        Retorna:
        - Acción seleccionada según la política ε-soft.
        """
        if self.decay:
            self.epsilon = min(1.0, 1000.0 / (n + 1))
        
        action_probabilities = np.ones(self.env.action_space.n) * (self.epsilon / self.env.action_space.n)
        best_action = np.argmax(self.Q[state])
        action_probabilities[best_action] += (1 - self.epsilon)
        
        return np.random.choice(self.env.action_space.n, p=action_probabilities)


    def update(self, episode: List[Tuple[int, int, float]]) -> None:
        """
        Actualiza la tabla Q(s, a) usando Monte Carlo Off-Policy con Importance Sampling.
        
        Parámetros:
        - episode: lista de tuplas (estado, acción, recompensa) que representan un episodio completo.
        """
        G = 0  # Retorno total acumulado
        W = 1  # Peso para el Importance Sampling

        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = self.gamma * G + reward  # Cálculo del retorno acumulado
            
            # Actualización ponderada de la tabla Q usando Importance Sampling
            self.C[state, action] += W
            self.Q[state, action] += (W / self.C[state, action]) * (G - self.Q[state, action])
            
            # Si la acción no es la óptima según la política objetivo, se detiene la actualización
            if action != np.argmax(self.Q[state]):
                break
            
            # Se actualiza W para la próxima iteración del Importance Sampling
            W = W * 1.0 / (self.epsilon / self.env.action_space.n + (1 - self.epsilon) * (action == np.argmax(self.Q[state])))
