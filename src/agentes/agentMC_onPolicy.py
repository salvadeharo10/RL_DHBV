import numpy as np
import gymnasium as gym
from typing import Any, List, Tuple
from agentes.agent import Agent

# Clase Monte Carlo On-Policy Agent
class MonteCarloOnPolicyAgent(Agent):
    """
    Implementación de un agente de Aprendizaje por Refuerzo basado en Monte Carlo On-Policy.
    Aprende una política óptima a partir de episodios completos y actualiza la función de valor Q(s, a).
    """
    def __init__(self, env, seed, gamma: float = 0.99, epsilon: float = 0.1, decay: bool = False):
        """
        Inicializa el agente con los parámetros de entrenamiento.
        
        Parámetros:
        - env: entorno de Gym donde se entrena el agente.
        - gamma: factor de descuento para ponderar recompensas futuras.
        - epsilon: parámetro para la política ε-soft, controlando la exploración.
        - decay: si es True, el valor de ε disminuye a medida que avanza el entrenamiento.
        """
        #super().__init__(env)
        self.env = env
        self.gamma = gamma  # Factor de descuento
        self.epsilon = epsilon  # Exploración epsilon-greedy
        self.decay = decay  # Control de decaimiento de epsilon
        
        # Tabla Q para almacenar valores de estado-acción
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        np.random.seed(self.seed)
        
        # Diccionario para almacenar retornos de cada par (estado, acción)
        # a lo largo de todos los episodios que dure el experimento
        self.returns = {}

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
            self.epsilon = min(1, 1000.0 / (n + 1))
        
        action_probabilities = np.ones(self.env.action_space.n) * (self.epsilon / self.env.action_space.n)
        best_action = np.argmax(self.Q[state])
        action_probabilities[best_action] += (1 - self.epsilon)
        
        return np.random.choice(self.env.action_space.n, p=action_probabilities)

    def update(self, episode: List[Tuple[int, int, float]]) -> None:
        """
        Actualiza la tabla Q(s, a) usando Monte Carlo On-Policy.
        
        Parámetros:
        - episode: lista de tuplas (estado, acción, recompensa) que representan un episodio completo.
        """
        states, actions, rewards = zip(*episode)
        G = 0

        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = self.gamma * G + reward  # Cálculo del retorno acumulado

            # Se actualiza solo si el par (estado, acción) aparece por primera vez en el episodio
            if (state, action) not in [(x[0], x[1]) for x in episode[:t]]:
                if (state, action) not in self.returns:
                    self.returns[(state, action)] = []
                
                self.returns[(state, action)].append(G)
                # Promedio de los retornos acumulados para actualizar Q(s, a)
                self.Q[state, action] = np.mean(self.returns[(state, action)])
