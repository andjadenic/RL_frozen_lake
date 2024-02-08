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
    return pi, q


if __name__ == '__main__':
    policy, q_value = q_learning(N_improvements=10000,
                                 alpha=0.1,
                                 epsilon=0.1,
                                 gamma=0.9)
    read_q_value(q_value)
    read_policy(policy)