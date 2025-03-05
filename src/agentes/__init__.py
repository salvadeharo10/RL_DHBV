# Importación de módulos o clases
from .agent import Agent
from .agentMC_onPolicy import MonteCarloOnPolicyAgent
from .agentMC_offPolicy import MonteCarloOffPolicyAgent
from .agentSARSA import AgentSarsa
from .agentQLearning import AgentQLearning
from .agentSarsaSemiGradient import SemiGradientSarsaAgent
from .dqnAgent import DQNAgent
from .agentQLearningCont import AgentQLearningCont


# Lista de módulos o clases públicas
__all__ = ['Agent', 'MonteCarloOnPolicyAgent', 'MonteCarloOffPolicyAgent', 'AgentSarsa', 'AgentQLearning', 'SemiGradientSarsaAgent', 'DQNAgent', 'AgentQLearningCont']
