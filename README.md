# RL_DHBV

## Información
- **Alumnos:**
  - de Haro Orenes, Salvador
  - Vidal García, Miguel
  - Belmonte Martínez, Jose María
 
## Descripción
En este repositorio se recopila el trabajo que tiene como objetivo abordar el problema del bandido multibrazo mediante el empleo de diferentes familias de algoritmos, tales como `Epsilon-greedy`, `UCB` (1 y 2) y `Métodos de Ascenso del Gradiente` (Softmax y Gradiente de preferencias).

## Estructura
- `src/` → Código Python relacionado con los experimentos: Implementación de los algoritmos y sus políticas; Implementación de los brazos con distintas distribuciones de probabilidad; Otras funcionalidades, tales como plotear gráficas.
- `docs/` → Documentación en PDF sobre el trabajo realizado.
- `tests/` → Scripts para pruebas automáticas.
- `data/` → Datos utilizados en los experimentos.
- `main.ipynb` → Notebook principal de presentación, que permite navegar entre los distintos notebooks con los experimentos.
- `1_bandit_experiments_epsilongreedy.ipynb` → Experimentos para la familia de algoritmos $\epsilon$-greedy.
- `2_bandit_experiments_ucb.ipynb` → Experimentos para la familia de algoritmos UCB, en particular UCB1 y UCB2.
- `3_bandit_experiments_ascensoGradiente.ipynb` → Experimentos para la familia de algoritmos de Ascenso del Gradiente, en particular Softmax y Gradiente de Preferencias.
- `4_bandit_experiments_torneoFinal.ipynb` → Comparativa final de los mejores algoritmos para las distintas distribuciones de probabilidad en el bandido.

## Instalación y uso
Se recomienda la ejecución de los notebooks en Google Colab. Para este fin, se aconseja abrir el notebook `main.ipynb`, donde se encontrarán enlaces directos para abrir los notebooks con los distintos experimentos en Colab.

## Tecnologías utilizadas
Python, GitHub y Google Colab.
