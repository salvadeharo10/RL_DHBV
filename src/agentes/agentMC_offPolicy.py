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
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        
        # Inicialización de las tablas Q y C (para Importance Sampling)
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        self.C = np.zeros((env.observation_space.n, env.action_space.n))
        self.policy = np.zeros(env.observation_space.n, dtype=int)


  def get_action(self, state: int, n: int) -> int:
    """
    Selecciona una acción siguiendo una política ε-greedy.
    
    Parámetros:
    - state: estado actual del agente.
    - n: número de episodios transcurridos (para el decaimiento de ε si está activado).
    
    Retorna:
    - Acción seleccionada según la política ε-greedy.
    """
    if self.decay:
        self.epsilon = min(1, 1000.0 / (n + 1))  # Ajuste dinámico de epsilon

    best_action = np.argmax(self.Q[state])  # Elegir la mejor acción según Q
    if np.random.rand() < self.epsilon:  # Con probabilidad ε, elegir acción aleatoria
        return np.random.choice(self.env.action_space.n)
    else:
        return best_action  # Con probabilidad (1 - ε), elegir la mejor acción



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

            # Actualización de la política óptima
            self.policy[state] = np.argmax(self.Q[state])  # π(St) ← argmax Q(St, a)
    
            # Si la acción tomada en el episodio no coincide con la nueva acción óptima, se detiene
            if action != self.policy[state]: # π(St) ← argmax Q(St, a)
                break
    
            # Actualización del factor de importancia W
            b_prob = self.epsilon / self.env.action_space.n + (1 - self.epsilon) * (action == self.policy[state])
            W = W / b_prob  # W ← W / b(A_t | S_t)

      
