
import gymnasium as gym
import numpy as np
from agentes import Agent  # Importamos la clase base Agent desde su módulo

class AgentQLearningCont(Agent):
    """
    Implementación de un agente Q-Learning.
    Aprende una política óptima actualizando su función Q(s, a) en cada paso de un episodio.
    """
    def __init__(self, env: gym.Env, gamma: float, epsilon: float, decay: bool, alpha: float):
        """
        Inicializa el agente con los parámetros de entrenamiento.
        
        Parámetros:
        - env: entorno de Gym donde se entrena el agente.
        - gamma: factor de descuento para ponderar recompensas futuras.
        - epsilon: parámetro para la política ε-soft, controlando la exploración.
        - decay: si es True, el valor de ε disminuye a medida que avanza el entrenamiento.
        - alpha: tasa de aprendizaje para actualizar la tabla Q.
        """
        super().__init__(env)
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        self.alpha = alpha  # Tasa de aprendizaje
        self.num_features = env.n_tilings * env.tile_size
        self.num_actions = self.env.action_space.n
        
        # Tabla Q para almacenar valores de estado-acción
        self.Q = np.zeros((self.num_features, self.num_actions))
    
    def get_q(self, active_features, action):
        """
        Calcula el valor de Q(s, a) como la suma de los pesos de las features activas.

        Parámetros:
        - active_features: Lista de índices de características activas en el estado s.
        - action: Acción para la cual se calcula Q(s, a).

        Retorna:
        - Valor de Q(s, a) sumando los pesos correspondientes a las características activas.
        """
        return np.sum(self.Q[active_features, action])
    
    def get_action(self, active_features, n):
        """
        Selecciona una acción siguiendo una política ε-soft.
        
        Parámetros:
        - active_feature: feature actual del agente.
        - n: número de episodios transcurridos (para el decaimiento de ε si está activado).
        
        Retorna:
        - Acción seleccionada según la política ε-soft.
        """
        if self.decay:
            self.epsilon = min(1.0, 1000.0 / (n + 1))
        
        action_probabilities = np.ones(self.env.action_space.n) * (self.epsilon / self.num_actions)
        q_values = np.array([self.get_q(active_features, a) for a in range(self.num_actions)])
        best_action = np.argmax(q_values)  # Selección de la mejor acción
      
        action_probabilities[best_action] += (1 - self.epsilon)
        
        return np.random.choice(self.env.action_space.n, p=action_probabilities)
    
    def update(self, active_feature: int, action: int, reward: float, next_active_feature: int, terminated: bool, truncated: bool) -> None:
        """
        Actualiza la tabla Q usando el algoritmo Q-Learning.
        
        Parámetros:
        - active_feature: feature actual.
        - action: acción tomada en el estado actual.
        - reward: recompensa obtenida tras tomar la acción.
        - next_active_feature: siguiente feature alcanzado.
        - terminated: indica si el episodio ha finalizado.
        - truncated: indica si el episodio ha sido truncado.
        """
        if terminated or truncated:
            # Si el episodio termina, solo se usa la recompensa inmediata para actualizar Q
            self.Q[active_feature, action] += self.alpha * (reward - self.Q[active_feature, action])
        else:
            # Q-Learning usa la mejor acción futura posible para actualizar Q
            self.Q[active_feature, action] += self.alpha * (
                reward + self.gamma * np.max(self.Q[next_active_feature]) - self.Q[active_feature, action]
            )
