import gymnasium as gym
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import uniform_filter1d
from agentes.agent import Agent

# Para grabar videosrom gymnasium.wrappers import RecordVideo

# Para renderizar correctamente en algunos entornos
import imageio
import moviepy.editor as mpy

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
    for episode in range(0, num_episodes, 1000):
        env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda eid: True)
        print(f"Grabando episodio {episode + 1}...")
        done = False
        pi_star = np.zeros([env.observation_space.n, env.action_space.n])
        state, info = env.reset()
        actions = f"Episodio {episode + 1}: "

        while not done:
            env.render()
            action = np.argmax(Q[state, :])
            actions += f"{action}, "
            pi_star[state, action] = action
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        print(f"Episodio {episode + 1} completado.")
        env.close()

    return pi_star, actions


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


def monte_carlo_on_policy(agent, env, num_episodes=5000, SEED):
    stats = 0.0
    list_stats = [stats]
    episode_lengths = []
    step_display = max(1, num_episodes // 10)

    for t in tqdm(range(num_episodes), disable=False):
        state, info = env.reset(seed=SEED)
        done = False
        episode = []
        result_sum = 0.0
        length = 0

        while not done:
            action = agent.get_action(state, t)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            done = terminated or truncated
            result_sum += reward
            length += 1

        agent.update(episode)
        episode_lengths.append(length)
        stats += result_sum
        list_stats.append(stats / (t + 1))

        if t % step_display == 0 and t != 0:
            print(f"Éxito promedio: {stats/t}, epsilon: {agent.epsilon:.4f}")

    return agent.Q, list_stats, episode_lengths


