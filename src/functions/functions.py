import gymnasium as gym
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import uniform_filter1d
from pyvirtualdisplay import Display
from gymnasium.wrappers import RecordVideo
import io
import base64
from IPython import display
from IPython.display import HTML

# Para renderizar correctamente en algunos entornos
import imageio

# Definición de acciones
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3

# Función para mostrar videos en el notebook
# Esta función mostrará un vídeos en la pantalla virtual, que será embembida en el notbook
def embed_video(video_file):
    # open and read the raw data from the video
    video_data = io.open(video_file, 'r+b').read()
    # now we have to encode the data into base64 to work
    # with the virtual display
    encoded_data = base64.b64encode(video_data)
    # now we use the display.display function to take some html
    # and the encoded data and embed the html into the notebook!
    display.display(HTML(data=''''''.format(encoded_data.decode('ascii'))))
    
# Función para obtener el último video generado
def get_latest_episode_video_file(directory):
    import re
    pattern = re.compile(r"rl-video-episode-(\d+)\.mp4")
    latest_file = None
    highest_episode = -1

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            episode_number = int(match.group(1))
            if episode_number > highest_episode:
                highest_episode = episode_number
                latest_file = os.path.join(directory, filename)

    return latest_file

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
    
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda eid: True)
    print(f"Grabando ejecución...")
    done = False
    pi_star = np.zeros([env.observation_space.n, env.action_space.n])
    state, info = env.reset()
    actions = ""
    frames = []  # Lista para almacenar cada fotograma.
    while not done:
        frame = env.render()
        frames.append(frame)
        action = np.argmax(Q[state, :])
        actions += f"{action}, "
        pi_star[state, action] = action
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    print(f"Grabación completada.")
    env.close()

    return pi_star, actions, frames

def display_gif(gif_path):
    """
    Muestra un GIF en Google Colab.

    Parámetros:
      - gif_path (str): Ruta del archivo GIF.

    Retorna:
      - HTML: Objeto HTML que contiene el GIF incrustado.
    """
    # Abrir el archivo GIF en modo binario.
    with open(gif_path, 'rb') as f:
        video = f.read()
    # Convertir el contenido del GIF a una cadena Base64.
    b64 = base64.b64encode(video)
    # Retornar el objeto HTML que muestra el GIF.
    return HTML(f'')

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


