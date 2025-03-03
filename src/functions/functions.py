import gymnasium as gym
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import uniform_filter1d
import io
import base64
from IPython import display
from IPython.display import HTML
import seaborn as sns

# Para renderizar correctamente en algunos entornos
import imageio

# Definición de acciones
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3

# Política Greedy a partir de los valores Q. Se usa para mostrar la solución.
def pi_star_from_Q(env, Q):
    done = False
    pi_star = np.zeros([env.observation_space.n, env.action_space.n])
    state, info = env.reset()
    actions = ""
    while not done:
        action = np.argmax(Q[state, :])
        actions += f"{action}, "
        pi_star[state, action] = action
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return pi_star, actions


def pi_star_from_Q_recordVideo(env, Q, video_folder="videos", num_episodes=5000):
    
    print(f"Grabando ejecución...")
    done = False
    pi_star = np.zeros([env.observation_space.n, env.action_space.n])
    state, info = env.reset()
    actions = ""
    frames = []  # Lista para almacenar cada fotograma.
    actions_list = []  # Lista de acciones tomadas
    visited_states = []  # Lista de estados visitados

    while not done:
        frame = env.render()
        frames.append(frame)
        action = np.argmax(Q[state, :])
        actions += f"{action}, "
        actions_list.append(action)  # Guardar acción
        pi_star[state, action] = action
        state, reward, terminated, truncated, info = env.step(action)
        visited_states.append(state)  # Guardar estado
        done = terminated or truncated

    print(f"Grabación completada.")
    env.close()

    return pi_star, actions, frames, actions_list, visited_states

def display_gif(gif_path):
    """
    Muestra un GIF en Google Colab o Jupyter Notebook.

    Parámetros:
      - gif_path (str): Ruta del archivo GIF.

    Retorna:
      - HTML: Objeto HTML que contiene el GIF incrustado.
    """
    # Abrir el archivo GIF en modo binario.
    with open(gif_path, "rb") as f:
        video = f.read()
    
    # Convertir el contenido del GIF a una cadena Base64 y decodificarlo como string
    b64 = base64.b64encode(video).decode("utf-8")  

    # Retornar el objeto HTML con el GIF embebido
    return HTML(f'<img src="data:image/gif;base64,{b64}" />')
    

def frames_to_gif(frames, filename="cartpole_sarsa.gif"):
    """
    Crea un archivo GIF a partir de una lista de fotogramas.

    Parámetros:
      - frames (list): Lista de fotogramas (imágenes) capturados del entorno.
      - filename (str): Nombre del archivo GIF resultante.

    Retorna:
      - str: Nombre del archivo GIF creado.
    """
    # Abrir un escritor de GIF con imageio.
    with imageio.get_writer(filename, mode='I') as writer:
        # Agregar cada fotograma al GIF.
        for frame in frames:
            writer.append_data(frame)
    return filename
    
    

def plot(list_stats):
    indices = list(range(len(list_stats)))
    plt.figure(figsize=(12, 6))
    plt.plot(indices, list_stats, label="Proporción de recompensas", color="blue")
    plt.title("Proporción acumulada de recompensas obtenidas")
    plt.xlabel("Episodio")
    plt.ylabel("Proporción de recompensas")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_episode_lengths(episode_lengths):
    indices = np.arange(1, len(episode_lengths) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(indices, episode_lengths, label="Longitud del episodio", alpha=0.7, color="blue")
    smoothed_lengths = uniform_filter1d(episode_lengths, size=50)
    plt.plot(indices, smoothed_lengths, label="Tendencia", linestyle="--", color="red")
    plt.title("Evolución de la longitud de los episodios")
    plt.xlabel("Episodio")
    plt.ylabel("Longitud del episodio")
    plt.legend()
    plt.grid(True)
    plt.show()

# Función para visualizar la distribución de estados y acciones
def plot_states_actions_distribution(states, actions, map_size):
    """Dibuja la distribución de estados visitados y acciones tomadas."""

    labels = {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3}

    # Convertir a listas si son arrays
    states = np.array(states).flatten()
    actions = np.array(actions).astype(int).flatten()

    # Crear figura con 2 gráficos
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    # Histograma de estados visitados
    sns.histplot(data=states, ax=ax[0], kde=True, bins=map_size**2, color="skyblue")
    ax[0].set_title("Distribución de Estados")
    ax[0].set_xlabel("Estados")
    ax[0].set_ylabel("Frecuencia")

    # Histograma de acciones tomadas
    sns.histplot(data=actions, ax=ax[1], bins=4, color="orange")
    ax[1].set_xticks(list(labels.values()))
    ax[1].set_xticklabels(labels.keys())
    ax[1].set_title("Distribución de Acciones")
    ax[1].set_xlabel("Acciones")
    ax[1].set_ylabel("Frecuencia")

    # Ajustar diseño
    fig.tight_layout()

    # Guardar la imagen
    img_title = f"frozenlake_states_actions_distrib_{map_size}x{map_size}.png"
    plt.savefig(img_title, bbox_inches="tight")
    plt.show()

def qtable_directions_map(qtable, map_size):
    """Convierte la Q-table en una matriz de valores y direcciones óptimas."""
    arrows = {0: "←", 1: "↓", 2: "→", 3: "↑"}  # Mapeo de acciones a flechas
    qtable_val_max = np.max(qtable, axis=1).reshape(map_size, map_size)  # Mejor valor Q por estado
    best_actions = np.argmax(qtable, axis=1).reshape(map_size, map_size)  # Mejor acción por estado
    qtable_directions = np.vectorize(arrows.get)(best_actions)  # Convertir acciones en flechas
    return qtable_val_max, qtable_directions

def plot_q_values_map(qtable, env, map_size):
    """Grafica el último frame de la simulación y la política aprendida."""
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, map_size)

    # Crear figura con dos subgráficos
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    # Última imagen del entorno
    last_frame = env.render()
    ax[0].imshow(last_frame)
    ax[0].axis("off")
    ax[0].set_title("Último frame del entorno")

    # Mapa de valores de Q con la política aprendida
    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="Valores de Q aprendidos\n(Flechas indican la mejor acción)")

    # Ajustes de los bordes
    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")

    # Guardar imagen
    img_title = f"frozenlake_q_values_{map_size}x{map_size}.png"
    plt.savefig(img_title, bbox_inches="tight")
    plt.show()
