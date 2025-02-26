from utils import *
import numpy as np
from collections import namedtuple
import random


episode = namedtuple('episode',
                     'states, actions, rewards, terminated, truncated')


def monte_carlo_control(N_episodes: int = 10, epsilon: float = 0.1, gamma: float = 0.9, is_slippery_bool: bool = True):
    q = np.zeros((16, 4))
    pi = .25 * np.ones((16, 4))

    for n_episode in range(1, N_episodes + 1):

        trajectory = sample_episode(pi=pi, is_slippery_bool=is_slippery_bool)

        G = 0
        for step in reversed(range(len(trajectory.states))):  # start from the last step
            s, a, r = trajectory.states[step], trajectory.actions[step], trajectory.rewards[step]

            first_visit = True
            for s_prev, a_prev in zip(trajectory.states[0:step-1], trajectory.actions[0:step-1]):
                if s_prev == s and a_prev == a:
                    first_visit = False

            if first_visit:
                G = r + gamma * G
                q[s, a] += (G - q[s, a]) / n_episode

                a_star = np.max(q[s, :])
                greedy_actions = []
                for i in range(4):
                    if q[s, i] == a_star:
                        greedy_actions.append(i)
                greedy_action = random.choice(greedy_actions)
                pi[s, :] = epsilon / 4
                pi[s, greedy_action] += 1 - epsilon
    pi = policy2greedy(pi=pi)
    return pi, q


if __name__ == '__main__':

    #  Monte Carlo Non-slippery Frozen Lake

    q_mc, pi_mc = monte_carlo_control(N_episodes=10000, epsilon=0.1, is_slippery_bool=False)
    # plot_result(pi_mc, 'Non-slipery Frozen Lake: MC policy')
    # plot_result(q_mc, 'Non-slipery Frozen Lake: MC q value')

    mean_return = calculate_mean_return(pi=pi_mc, N_runs=50000, gamma=0.9, is_slippery_bool=False)
    print(f'Slipery Frozen Lake:  MC {mean_return=}')


    #  Monte Carlo Slippery Frozen Lake

    q_mc, pi_mc = monte_carlo_control(N_episodes=10000, epsilon=0.1, is_slippery_bool=True)
    # plot_result(pi_mc, 'Slipery Frozen Lake: MC policy')
    # plot_result(q_mc, 'Slipery Frozen Lake: MC q value')

    mean_return = calculate_mean_return(pi=pi_mc, N_runs=50000, gamma=0.9, is_slippery_bool=True)
    print(f'Slipery Frozen Lake:  MC {mean_return=}')
