import gymnasium as gym
import numpy as np
from utils import plot_result
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import random


def policy_evaluation(P: dict, pi: np.ndarray, gamma: float = .9, theta: float = 1e-8):

    v = np.zeros((16,))
    q = np.zeros((15, 4))

    # in-place policy evaluation
    max_diff = 1e10
    while max_diff > theta:
        max_diff = 0
        for s in reversed(range(15)):
            v_prev = v[s]
            for a in range(4):
                for prob, s_new, r, _ in P[s][a]:
                    q[s][a] = prob * (r + gamma * v[s_new])
                v[s] = np.sum(pi[s, :] * q[s, :])
            max_diff = max(abs(v[s] - v_prev), max_diff)
    return v, q


if __name__ == '__main__':

    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
    P = env.P
    env.close()

    # initialize policy pi
    pi = 0.25 * np.ones((16, 4))

    # initialize hyperparameters
    theta = 1e-8
    gamma = 0.9

    converge = False
    while not converge:
        # 1. policy evaluation
        v, q = policy_evaluation(P, pi, gamma=gamma, theta=theta)

        # 2. policy improvement
        pi_prim = np.zeros((16, 4))
        for s in range(15):
            max_el = np.max(q[s, :])
            greedy_actions = []
            for a in range(4):
                if q[s][a] == max_el:
                    greedy_actions.append(a)
            pi_prim[s, greedy_actions] = 1 / len(greedy_actions)

        # 3. stop if v converged
        if np.max(abs(policy_evaluation(P, pi)[0] - policy_evaluation(P, pi_prim)[0])) < theta * 1e2:
            converge = True

        # 4. Replace policy with new policy
        pi = copy.copy(pi_prim)

    plot_result(q, 'q value')
    plt.figure(figsize=(8, 8))
    sns.heatmap(v.reshape(4, 4),  cmap="YlGnBu", annot=True, cbar=False, square=True)
    plot_result(pi, 'policy pi')
    plt.show()
