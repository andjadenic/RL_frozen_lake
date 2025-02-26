import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import gymnasium as gym
import random


episode = namedtuple('episode',
                     'states, actions, rewards, terminated, truncated')


def sample_episode(pi: np.ndarray = 0.25 * np.ones((16, 4)), is_slippery_bool: bool = True) -> episode:
    '''
    function simulates a single episode of a game following the given policy pi

    INPUTS
    pi: (16, 4) numpy array
        - pi[s, a] is probability of agent taking action a given he is in state s
        - for given s: we choose actions 0, 1, 2, 3 with probabilities pi[s, 0], pi[s, 1], pi[s, 2], pi[s, 3], respectively

    OPUTPUTS:
    episode: named tuple that collects trajectory information {s0, a0, r0, ..., s_T, A_T, R_T}
        - single episode has form: episode(states=[0, 1, 2], actions=[1, 2, 3], rewards=[0, 0, 1], terminated=True, truncated=False)
    '''
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=is_slippery_bool)

    s, info = env.reset(seed=42)
    terminated, truncated = False, False
    states, actions, rewards = [], [], []
    while not truncated and not terminated:
        a = np.random.choice([0, 1, 2, 3], p=pi[s, :])
        s_new, r, terminated, truncated, _ = env.step(a)
        states.append(s)
        actions.append(a)
        rewards.append(r)
        s = s_new
    env.close()
    return episode(states, actions, rewards, terminated, truncated)


def plot_result(matrix: np.ndarray, text: str):
    # Define actions and their corresponding arrow directions
    actions = ['left', 'down', 'right', 'up']
    dx = [-1, 0, 1, 0]
    dy = [0, -1, 0, 1]

    # Create a 4x4 plot
    fig, axs = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(f'{text}', ha='center', fontsize=16)

    # Iterate through the matrix and plot arrows
    for idx, cell in enumerate(matrix):
        i, j = divmod(idx, 4)
        max_prob = np.max(cell)
        for action, p in enumerate(cell):
            color = 'red' if p == max_prob else 'black'
            axs[i, j].arrow(0.5, 0.5, dx[action] * 0.2 * p, dy[action] * 0.2 * p, head_width=0.05, head_length=0.1,
                            fc=color, ec=color)
            axs[i, j].text(0.5 + dx[action] * 0.3, 0.5 + dy[action] * 0.3, f'{p:.2f}', color=color, fontsize=12)
        axs[i, j].set_xlim(0, 1)
        axs[i, j].set_ylim(0, 1)
        axs[i, j].axis('off')
    plt.show()


def calculate_mean_return(pi: np.ndarray,
                          N_runs: int = 50000, gamma: float = 0.9,
                          is_slippery_bool: bool = False):
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=is_slippery_bool)
    mean_return = 0
    s, _ = env.reset(seed=42)
    for n_run in range(N_runs):
        terminated, truncated = False, False
        discount_factor = 1
        while not terminated and not truncated:
            a = np.argmax(pi[s, :])
            new_s, r, terminated, truncated, _ = env.step(a)
            mean_return += discount_factor * r
            discount_factor *= gamma
            s = new_s
        s, _ = env.reset()
    env.close()
    mean_return /= N_runs
    return mean_return


def policy2greedy(pi: np.ndarray):
    greedy_policy = np.zeros_like(pi)
    for s in range(15):
        max_val = np.max(pi[s, :])
        greedy_actions = []
        for a in range(4):
            if pi[s, a] == max_val:
                greedy_actions.append(a)
        greedy_policy[s, greedy_actions] = 1 / len(greedy_actions)
    return greedy_policy
