import numpy as np
import gymnasium as gym
from agent import Agent  # Importamos la clase base Agent desde su módulo

class SemiGradientSarsaAgent(Agent):
    """
    Implementación de un agente SARSA con gradiente semilineal.
    Utiliza Tile Coding para representar estados y actualizar los pesos de manera eficiente.
    """
    def __init__(self, step_size: float, action_space: np.ndarray, env_tile, epsilon, decay):
        """
        Inicializa el agente con los parámetros de entrenamiento.
        
        Parámetros:
        - step_size: tasa de aprendizaje para la actualización de los pesos.
        - action_space: conjunto de acciones disponibles.
        - env_tile: objeto de codificación en mosaicos (Tile Coding) para representar estados.
        - epsilon: parámetro para la política ε-soft, controlando la exploración.
        - decay: si es True, el valor de ε disminuye a medida que avanza el entrenamiento.
        """
        self.env_tile = env_tile
        self.action_space = action_space
        self.alpha = step_size / self.env_tile.get_number_tilings()
        self.epsilon = epsilon
        self.decay = decay
        
        # Inicialización de los pesos para la aproximación de la función de valor
        self.weights = np.zeros(self.env_tile.get_maximum_tiles())

    def update_weights(self, state: np.ndarray, action: int, target: float) -> None:
        """
        Actualiza los pesos del modelo usando el error TD (Diferencia Temporal).
        
        Parámetros:
        - state: estado actual del agente.
        - action: acción tomada en el estado actual.
        - target: valor objetivo basado en la recompensa y el valor estimado futuro.
        """
        features = self.env_tile.get_features(state, action)
        estimated_value = np.sum(self.weights[features])
        td_error = (target - estimated_value)
        self.weights[features] += self.alpha * td_error

    def estimate_value(self, state: np.ndarray, action: int) -> float:
        """
        Estima el valor del estado-acción utilizando la suma ponderada de los pesos.
        
        Parámetros:
        - state: estado actual.
        - action: acción seleccionada.
        
        Retorna:
        - Estimación del valor Q(s,a) para el estado y acción dados.
        """
        features = self.env_tile.get_features(state, action)
        estimated_value = np.sum(self.weights[features])
        return estimated_value

    def get_greedy_action(self, state: np.ndarray) -> int:
        """
        Selecciona la mejor acción posible (política greedy) en función de los valores estimados.
        
        Parámetros:
        - state: estado actual del agente.
        
        Retorna:
        - Acción con el mayor valor estimado de Q(s,a).
        """
        values = [self.estimate_value(state, a) for a in self.action_space]
        values = np.array(values)
        best_action_idx = np.argmax(np.random.random(values.shape) * (values == values.max()))
        return self.action_space[best_action_idx]

    def get_epsilon_greedy_action(self, state: np.ndarray, n:int) -> int:
        """
        Selecciona una acción utilizando una política ε-greedy.
        
        Parámetros:
        - state: estado actual del agente.
        - n: numero de episodios transcurridos para decaimiento de epsilon
        
        Retorna:
        - Acción seleccionada de manera ε-greedy.
        """

        if self.decay:
            self.epsilon = min(1.0, 1000.0 / (n + 1))
                               
        if np.random.random() < 1 - self.epsilon:
            return self.get_greedy_action(state)  # Explotación
        else:
            return np.random.choice(self.action_space)  # Exploración
