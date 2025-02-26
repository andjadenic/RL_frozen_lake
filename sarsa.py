<<<<<<< HEAD
from utils import *
import random


def sarsa_control(expected_sarsa: bool = False,
                  N_episodes: int = 1000,
                  alpha: float = 0.1, epsilon: float = 0.1, gamma: float = 0.9,
                  is_slippery_bool: bool = False):
    q = np.zeros((16, 4))
    pi = .25 * np.ones((16, 4))
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=is_slippery_bool)
    s, info = env.reset(seed=42)

    for n_episode in range(1, N_episodes + 1):
        a = np.random.choice([0, 1, 2, 3], p=pi[s, :])
        terminated, truncated = False, False

        while not terminated and not truncated:
            # take action a and observe reward r, following state s_new (terminated, truncated)
            s_new, r, terminated, truncated, _ = env.step(a)

            # sample action a_new from s_new following current policy
            a_new = np.random.choice([0, 1, 2, 3], p=pi[s_new, :])

            if not expected_sarsa:
                q[s, a] += alpha * (r + gamma * q[s_new, a_new] - q[s, a])
            else:
                expected_q = 0
                for action in range(4):
                    expected_q += pi[s_new, action] * q[s_new, action]
                q[s, a] += alpha * (r + gamma * expected_q - q[s, a])

            # update policy pi for state s
            a_star = np.max(q[s, :])
            greedy_actions = []
            for i in range(4):
                if q[s, i] == a_star:
                    greedy_actions.append(i)
            greedy_action = random.choice(greedy_actions)
            pi[s, :] = epsilon / 4
            pi[s, greedy_action] += 1 - epsilon

            s = s_new
            a = a_new

        s, info = env.reset()

    env.close()
    pi = policy2greedy(pi)
=======
'''
SARSA ALORITHM:

INPUTS:
- alpha (step size, small float number from (0, 1])
- gamma (discount factor, float number close to 1 from [0, 1])
- N_improvements
- epsilon (small float number from (0, 1] that determine how much exploration we do)

INITIALIZATION:
policy = random policy
q[s, a] = 0, for every state-action pair

for n in range(N_improvements):
    1. sarsa prediction: estimate q
        s = 0
        sample action a following current policy
        terminated, truncated = False, False
        while not terminated and not truncated do:
            take action a and observe reward r, following state s` (terminated, truncated)
            sample action a` following current policy
            update q[s, a] using TD(0) formula:
                q[s, a] = q[s, a] + alpha * (r + gamma * q[s`, a`] - q[s, a])
            s = s`, a = a`

    2. sarsa control: improve policy
       use epsilon-greedy policy improvement derived from q

OUTPUT policy
'''
import numpy as np
import gymnasium as gym
from utils import epsolon_greedy_policy_improvement, read_policy, read_q_value


def sarsa_control(N_improvements: int = 10, alpha:float = 0.1, epsilon: float = 0.1, gamma: float = 0.9):
    q = np.zeros((16, 4))
    pi = .25 * np.ones((4, 16))
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    s, info = env.reset(seed=42)

    for n in range(N_improvements):
        #  1. sarsa prediction: estimate q

        # sample action a following current policy
        a = np.random.choice([0, 1, 2, 3], p=pi[:, s])
        a = int(a)
        terminated, truncated = False, False

        while not terminated and not truncated:
            # take action a and observe reward r, following state s` (terminated, truncated)
            s_prim, r, terminated, truncated, _ = env.step(a)

            # sample action a` following current policy
            a_prim = np.random.choice([0, 1, 2, 3], p=pi[:, s_prim])
            a_prim = int(a_prim)

            # update q[s, a]
            q[s, a] = q[s, a] + alpha * (r + gamma * q[s_prim, a_prim] - q[s, a])

            s = s_prim
            a = a_prim

            if terminated or truncated:
                s, info = env.reset()

        # 2. sarsa control: improve policy
        pi = epsolon_greedy_policy_improvement(q_value=q, epsilon=epsilon)

    env.close()
>>>>>>> f7069eda8304abfc62a611cbf550760d9009a56f
    return pi, q


if __name__ == '__main__':
<<<<<<< HEAD

    #  SARSA Non-slippery Frozen Lake

    q_sarsa, pi_sarsa = sarsa_control(expected_sarsa=False, N_episodes=10000, epsilon=0.1, is_slippery_bool=False)

    # plot_result(pi_sarsa, 'Non-slipery Frozen Lake: SARSA policy')
    # plot_result(q_sarsa, 'Non-slipery Frozen Lake: SARSA q value')

    mean_return = calculate_mean_return(pi=pi_sarsa, N_runs=50000, gamma=0.9, is_slippery_bool=False)
    print(f'Non-slipery Frozen Lake:  SARSA {mean_return=}')


    #  SARSA Slippery Frozen Lake

    q_sarsa, pi_sarsa = sarsa_control(expected_sarsa=False, N_episodes=10000, epsilon=0.1, is_slippery_bool=True)

    # plot_result(pi_sarsa, 'Slipery Frozen Lake: SARSA policy')
    # plot_result(q_sarsa, 'Slipery Frozen Lake: SARSA q value')

    mean_return = calculate_mean_return(pi=pi_sarsa, N_runs=50000, gamma=0.9, is_slippery_bool=True)
    print(f'Slipery Frozen Lake:  SARSA {mean_return=}')


    #  Expected  SARSA Non-slippery Frozen Lake

    q_exp_sarsa, pi_exp_sarsa = sarsa_control(expected_sarsa=True, N_episodes=10000, epsilon=0.1, is_slippery_bool=False)

    # plot_result(pi_sarsa, 'Non-slipery Frozen Lake: Expected SARSA policy')
    # plot_result(q_sarsa, 'Non-slipery Frozen Lake: Expected SARSA q value')

    mean_return = calculate_mean_return(pi=pi_sarsa, N_runs=50000, gamma=0.9, is_slippery_bool=False)
    print(f'Non-slipery Frozen Lake:  Expected SARSA {mean_return=}')


    #  Expected SARSA Slippery Frozen Lake

    q_exp_sarsa, pi_exp_sarsa = sarsa_control(expected_sarsa=True, N_episodes=10000, epsilon=0.1, is_slippery_bool=True)

    # plot_result(pi_sarsa, 'Slipery Frozen Lake: Expected SARSA policy')
    # plot_result(q_sarsa, 'Slipery Frozen Lake: Expected SARSA q value')

    mean_return = calculate_mean_return(pi=pi_sarsa, N_runs=50000, gamma=0.9, is_slippery_bool=True)
    print(f'Slipery Frozen Lake:  Expected SARSA {mean_return=}')
=======
    policy, q_value = sarsa_control(N_improvements=100000,
                                     alpha=0.1,
                                     epsilon=0.1,
                                     gamma=0.9)
    read_q_value(q_value)
    read_policy(policy)
>>>>>>> f7069eda8304abfc62a611cbf550760d9009a56f
