import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import gymnasium as gym
import random

<<<<<<< HEAD

=======
>>>>>>> f7069eda8304abfc62a611cbf550760d9009a56f
episode = namedtuple('episode',
                     'states, actions, rewards, terminated, truncated')


<<<<<<< HEAD
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
=======
def sample_episodes(pi: np.ndarray = 0.25 * np.ones((4, 16)), N_episodes: int = 1, print_steps: bool = False) -> list:
    '''
    function simulates n_ep number of games that follows the given policy pi

    INPUTS
    pi: (4, 16) numpy array
        - pi[a, s] is probability of agent taking action a from state s
        - for given s: we choose actions 0, 1, 2, 3 with probabilities pi[0, s], pi[1, s], pi[2, s], pi[3, s], respectively
    N_episodes: number of episodes we want to simulate
    eps: parameter in epsilon-greedy action selection
    print_steps: print_steps=True if we want to print each step

    OPUTPUTS:
    h - list of n_ep numed tuples that collects all trajectories {s0, a0, r0, ...}
        - single trajectory in a list has form episode(states=[0, 1, 2], actions=[1, 2, 3], rewards=[0, 0, 1])
    '''
    #  env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode='human')
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    h = []
    s, info = env.reset(seed=42)
    for n_episode in range(1, N_episodes + 1):
        terminated, truncated = False, False
        step = 0
        states, actions, rewards = [], [], []

        while not truncated and not terminated:
            a = np.random.choice([0, 1, 2, 3], p=pi[:, s])
            a = int(a)
            s_new, r, terminated, truncated, _ = env.step(a)
            if print_steps:
                print(f'e{n_episode} s{step}  :  ({s}, {arrows[a]}, {r}, {terminated}, {truncated})')
            states.append(s)
            actions.append(a)
            rewards.append(r)
            if terminated or truncated:
                s, _ = env.reset()
                if print_steps:
                    print('RESET')
            else:
                s = s_new
            step += 1

            if terminated or truncated:
                s, info = env.reset()

        h.append(episode(states, actions, rewards, terminated, truncated))
    env.close()
    return h


def read_policy(pi: np.ndarray, non_greedy_arrows: bool = True, non_greedy_prob: bool = True):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Set cell width and height
    cell_width = 0.25
    cell_height = 0.25

    # Loop through each cell
    for r in range(4):
        for c in range(4):
            # Calculate coordinates for the center of the cell
            x_center = c * cell_width + cell_width / 2 + c * 0.2
            y_center = (3 - r) * cell_height + cell_height / 2 - r * 0.2

            # Draw left arrow
            ax.arrow(
                x_center,
                y_center,
                -cell_width / 4,
                0,
                head_width=0.05,
                head_length=0.05,
                length_includes_head=False,
                color="black",
            )

            # Draw down arrow
            ax.arrow(
                x_center,
                y_center,
                0,
                -cell_height / 4,
                head_width=0.05,
                head_length=0.05,
                length_includes_head=False,
                color="black",
            )

            # Draw right arrow
            ax.arrow(
                x_center,
                y_center,
                cell_width / 4,
                0,
                head_width=0.05,
                head_length=0.05,
                length_includes_head=False,
                color="black",
            )

            # Draw up arrow
            ax.arrow(
                x_center,
                y_center,
                0,
                cell_height / 4,
                head_width=0.05,
                head_length=0.05,
                length_includes_head=False,
                color="black",
            )

            # Get the values for the current cell
            values = pi[:, 3 * r + c]

            # Write the values next to the arrows
            ax.text(
                x_center - cell_width / 4 - 0.05,
                y_center,
                f"{values[0]:.2f}",
                ha="right",
                va="center",
                fontsize=8,
            )
            ax.text(
                x_center,
                y_center - cell_height / 4 - 0.1,
                f"{values[1]:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            ax.text(
                x_center + cell_width / 4 + 0.05,
                y_center,
                f"{values[2]:.2f}",
                ha="left",
                va="center",
                fontsize=8,
            )
            ax.text(
                x_center,
                y_center + cell_height / 4 + 0.1,
                f"{values[3]:.2f}",
                ha="center",
                va="top",
                fontsize=8,
            )

    # Set axis limits and labels
    ax.set_xlim(-0.08, 1.7)
    ax.set_ylim(-0.7, 1.1)
    ax.set_xticks([])
    ax.set_yticks([])

    # Set title
    ax.set_title("Policy and greedy-actions")

    plt.show()


def read_q_value(q: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Set cell width and height
    cell_width = 0.25
    cell_height = 0.25

    # Loop through each cell
    for r in range(4):
        for c in range(4):
            # Calculate coordinates for the center of the cell
            x_center = c * cell_width + cell_width / 2 + c * 0.2
            y_center = (3 - r) * cell_height + cell_height / 2 - r * 0.2

            # Draw left arrow
            ax.arrow(
                x_center,
                y_center,
                -cell_width / 4,
                0,
                head_width=0.05,
                head_length=0.05,
                length_includes_head=False,
                color="black",
            )

            # Draw down arrow
            ax.arrow(
                x_center,
                y_center,
                0,
                -cell_height / 4,
                head_width=0.05,
                head_length=0.05,
                length_includes_head=False,
                color="black",
            )

            # Draw right arrow
            ax.arrow(
                x_center,
                y_center,
                cell_width / 4,
                0,
                head_width=0.05,
                head_length=0.05,
                length_includes_head=False,
                color="black",
            )

            # Draw up arrow
            ax.arrow(
                x_center,
                y_center,
                0,
                cell_height / 4,
                head_width=0.05,
                head_length=0.05,
                length_includes_head=False,
                color="black",
            )

            # Get the values for the current cell
            values = q[3 * r + c, :]

            # Write the values next to the arrows
            ax.text(
                x_center - cell_width / 4 - 0.05,
                y_center,
                f"{values[0]:.2f}",
                ha="right",
                va="center",
                fontsize=8,
            )
            ax.text(
                x_center,
                y_center - cell_height / 4 - 0.1,
                f"{values[1]:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            ax.text(
                x_center + cell_width / 4 + 0.05,
                y_center,
                f"{values[2]:.2f}",
                ha="left",
                va="center",
                fontsize=8,
            )
            ax.text(
                x_center,
                y_center + cell_height / 4 + 0.1,
                f"{values[3]:.2f}",
                ha="center",
                va="top",
                fontsize=8,
            )

    # Set axis limits and labels
    ax.set_xlim(-0.08, 1.7)
    ax.set_ylim(-0.7, 1.1)
    ax.set_xticks([])
    ax.set_yticks([])

    # Set title
    ax.set_title("q(state, action) values")

    plt.show()


def epsolon_greedy_policy_improvement(q_value: np.ndarray, epsilon: float = 0.1):
    """
    function does single update on policy using q_value and
    epsilon-greedy (greedy when epsilon == 0) policy improvement

    INPUTS
    q_value: (4, 16) numpy array
            -> q_value[a, s] is probability of agent taking action a from state s
    epsilon: float number from [0, 1] determining how much policy is greedy
             -> probability for taking argmax action is 1 - epsilon
                and probability of taking other actions is equal and adds up to epsilon
             -> when epsilon == 0, policy is greedy
             -> when epsilon == 1, policy is random (uniform)

    OUTPUT
    policy: improved policy
    """
    policy = np.zeros((4, 16))

    # policy improvement
    for state in range(16):
        policy[:, state] = epsilon / 3  # non-greedy actions have probability of selection = epsilon / 3

        greedy_actions = np.where(q_value[state, :] == np.max(q_value[state, :]))
        greedy_actions = greedy_actions[0]
        greedy_action = random.choice(greedy_actions)
        greedy_action = int(greedy_action)
        policy[greedy_action, state] = 1 - epsilon  # greedy actions have probability of selection = 1 - epsilon
    return policy


if __name__=='__main__':
    pi = np.random.rand(4, 16)  # Create a random NumPy ndarray
    #read_policy(pi)

    q = np.random.rand(16, 4)
    read_q_value(q=q)
>>>>>>> f7069eda8304abfc62a611cbf550760d9009a56f
