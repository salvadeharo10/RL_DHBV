
import gymnasium as gym
import numpy as np
from agentes import Agent  # Importamos la clase base Agent desde su módulo

class AgentSarsaCont(Agent):
    """
    Implementación de un agente SARSA (State-Action-Reward-State-Action) para espacion continuos.
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
        
        # Tabla Q para almacenar valores de estado-acción
        self.num_features = env.n_tilings * np.prod(env.bins)
        self.Q = np.zeros((self.num_features , self.env.action_space.n))

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
        - state: estado actual del agente.
        - n: número de episodios transcurridos (para el decaimiento de ε si está activado).
        
        Retorna:
        - Acción seleccionada según la política ε-soft.
        """
        if self.decay:
            self.epsilon = min(1.0, 1000.0 / (n + 1))
        
        action_probabilities = np.ones(self.env.action_space.n) * (self.epsilon / self.env.action_space.n)
        q_values = np.array([self.q_value(active_features, a) for a in range(self.num_actions)])
        best_action = np.argmax(q_values)  # Selección de la mejor acción
      
        action_probabilities[best_action] += (1 - self.epsilon)
        
        return np.random.choice(self.env.action_space.n, p=action_probabilities)
    
    def update(self, active_feature: int, action: int, reward: float, next_active_feature: int, next_action: int, terminated: bool, truncated: bool) -> None:
        """
        Actualiza la tabla Q usando el algoritmo SARSA.
        
        Parámetros:
        - state: estado actual del agente.
        - action: acción tomada en el estado actual.
        - reward: recompensa obtenida tras tomar la acción.
        - next_state: siguiente estado alcanzado.
        - next_action: siguiente acción tomada en el nuevo estado.
        - terminated: indica si el episodio ha finalizado.
        - truncated: indica si el episodio ha sido truncado.
        """
        if terminated or truncated:
            # Si el episodio termina, solo se usa la recompensa inmediata para actualizar Q
            self.Q[active_feature, action] += self.alpha * (reward - self.Q[active_feature, action])
        else:
            # SARSA utiliza la siguiente acción tomada para actualizar Q en base a la política actual
            self.Q[active_feature, action] += self.alpha * (
                reward + self.gamma * self.Q[next_active_feature, next_action] - self.Q[active_feature, action]
            )
