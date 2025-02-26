<<<<<<< HEAD
from utils import *


def q_learning(N_episodes: int = 1000,
               alpha: float = 0.01, epsilon: float = 0.1, gamma: float = 0.9,
               is_slippery_bool: bool = False):
    q = np.zeros((16, 4))
    pi = .25 * np.ones((16, 4))
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=is_slippery_bool)
    s, info = env.reset(seed=42)

    for n_episode in range(1, N_episodes + 1):
        terminated, truncated = False, False

        while not terminated and not truncated:
            # sample action a from state s following policy pi
            a = np.random.choice([0, 1, 2, 3], p=pi[s, :])

            # take action a and observe reward r, following state s_new (terminated, truncated)
            s_new, r, terminated, truncated, _ = env.step(a)

            # update q[s, a]
            q_max = np.max(q[s_new, :])
            greedy_actions = []
            for action in range(4):
                if q[s_new, action] == q_max:
                    greedy_actions.append(action)
            a_new_star = random.choice(greedy_actions)
            q[s, a] = q[s, a] + alpha * (r + gamma * q[s_new, a_new_star] - q[s, a])

            # update policy pi for state s
            a_star = np.max(q[s, :])
            greedy_actions = []
            for action in range(4):
                if q[s, action] == a_star:
                    greedy_actions.append(action)
            greedy_action = random.choice(greedy_actions)
            pi[s, :] = epsilon / 4
            pi[s, greedy_action] += 1 - epsilon

            s = s_new

        s, info = env.reset()

    env.close()
    pi = policy2greedy(pi)
=======
'''
Q-learning ALORITHM:

INPUTS:
- alpha (step size, small float number from (0, 1])
- gamma (discount factor, float number close to 1 from [0, 1])
- N_improvements
- epsilon (small float number from (0, 1] that determine how much exploration we do)

INITIALIZATION:
policy = random policy
q[s, a] = 0, for every state-action pair

for n in range(N_improvements):
    1. q-learning prediction: estimate q
        s = 0
        terminated, truncated = False, False
        while not terminated and not truncated do:
            sample action a following current policy
            take action a and observe reward r, following state s` (terminated, truncated)
            update q[s, a] using q-learning update formula:
                q[s, a] = q[s, a] + alpha * (r + gamma * max_a'(q[s`, a`]) - q[s, a])
            s = s'

    2. q-learning control: improve policy
       use epsilon-greedy policy improvement derived from q

OUTPUT policy
'''
import numpy as np
import gymnasium as gym
from utils import epsolon_greedy_policy_improvement, read_policy, read_q_value


def q_learning(N_improvements: int = 10, alpha:float = 0.1, epsilon: float = 0.1, gamma: float = 0.9):
    q = np.zeros((16, 4))
    pi = .25 * np.ones((4, 16))
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    s, info = env.reset(seed=42)

    for n in range(N_improvements):
        #  1. q-learning prediction: estimate q
        terminated, truncated = False, False

        while not terminated and not truncated:
            a = np.random.choice([0, 1, 2, 3], p=pi[:, s])
            a = int(a)
            s_prim, r, terminated, truncated, _ = env.step(a)

            # update q[s, a]
            q[s, a] = q[s, a] + alpha * (r + gamma * np.max(q[s_prim, :]) - q[s, a])

            s = s_prim

        s, info = env.reset()

        # 2. sarsa control: improve policy
        pi = epsolon_greedy_policy_improvement(q_value=q, epsilon=epsilon)

    env.close()
>>>>>>> f7069eda8304abfc62a611cbf550760d9009a56f
    return pi, q


if __name__ == '__main__':
<<<<<<< HEAD

    # Non-slippery Frozen Lake

    pi_q_learning, q_q_learning = q_learning(N_episodes=10000, is_slippery_bool=False)
    # plot_result(q_q_learning, 'Non-slippery Frozen Lake: Q-Learning q value')
    # plot_result(pi_q_learning, 'Non-slippery Frozen Lake: Q-Learning q value')

    mean_return = calculate_mean_return(pi_q_learning, N_runs=50000, is_slippery_bool=False)
    print(f'Non-slipery Frozen Lake:  Q-Learning {mean_return=}')

    # Slippery Frozen Lake

    pi_q_learning, q_q_learning = q_learning(N_episodes=10000, is_slippery_bool=True)
    # plot_result(q_q_learning, 'Slippery Frozen Lake: Q-Learning q value')
    # plot_result(pi_q_learning, 'Slippery Frozen Lake: Q-Learning q value')

    mean_return = calculate_mean_return(pi_q_learning, N_runs=50000, is_slippery_bool=True)
    print(f'Slipery Frozen Lake:  Q-Learning {mean_return=}')
=======
    policy, q_value = q_learning(N_improvements=10000,
                                 alpha=0.1,
                                 epsilon=0.1,
                                 gamma=0.9)
    read_q_value(q_value)
    read_policy(policy)
>>>>>>> f7069eda8304abfc62a611cbf550760d9009a56f
