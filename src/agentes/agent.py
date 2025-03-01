from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym
from typing import Any

class Agent(ABC):
    def __init__(self, env: gym.Env):
        """Inicializa todo lo necesario para el aprendizaje.
        : param env: entorno gymnasium
        """
        self.env = env

    @abstractmethod
    def get_action(self, state: Any) -> Any:
        """Indicará qué acción realizar de acuerdo al estado.
        Responde a la política del agente.
        Construir tantas funciones de este tipo como políticas se quieran usar.
        """
        raise NotImplementedError("Este método debe ser implementado por la subclase.")

    @abstractmethod
    def update(self, state: Any, action: Any, next_state: Any, reward: float, terminated: bool, truncated: bool, info: dict) -> None:
        """Con la muestra (s, a, s', r) e información complementaria aplicamos el algoritmo.
        update() no es más que el algoritmo de aprendizaje del agente.
        Se añadirá lo necesario para obtener resultados estadísticos, evolución, etc.
        """
        raise NotImplementedError("Este método debe ser implementado por la subclase.")