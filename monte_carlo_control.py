<<<<<<< HEAD
from utils import *
import numpy as np
from collections import namedtuple
import random


=======
"""
MONTE CARLO CONTROL

- we know how to calculate state-action value function for any policy
  using monte carlo policy evaluation
- goal in MC Control stage is to calculate optimal policy
- in dynamic programing for finding optimal policy we used bellman's equation
  that uses transitional probabilities
- here, policy improvement is done by updating policy following:
        EPSILON-GREEDY POLICY UPDATE with respect to the current state-action value function

- we don't need the model for these improvements
- for policy improvement we use knowledge about current policy in these Monte Carlo Reinforcement Learning methods.
  They are called ON POLICY METHODS
- there are also OFF POLICY METHODS studied the next file


ALGORITHM for Monte Carlo control:
inputs: epsilon, N_episodes, discount
initialize:
    - policy to be random
    - q-value function to have value 0 for each state-action pair
for N_episodes repeat:
    1. sample an episode {s_0, a_0, r_0, ..., s_n, a_n, r_n} following the current policy
    2. Monte Carlo Prediction stage:
           estimate state-action value function using sampled episode
    3. Monte Carlo Control stage:
           update policy using epsilon-greedy method
           (for each state s action with the highest q-value will have probability 1-epsilon
           and other actions will have equal probabilities that sums up to epsilon)
"""
from utils import read_policy, read_q_value, epsolon_greedy_policy_improvement, sample_episodes
import numpy as np
import gymnasium as gym
from collections import namedtuple


arrows = [u'\u2190', u'\u2193', u'\u2192', u'\u2191']  # 0-left, 1-down, 2-right, 3-up
>>>>>>> f7069eda8304abfc62a611cbf550760d9009a56f
episode = namedtuple('episode',
                     'states, actions, rewards, terminated, truncated')


<<<<<<< HEAD
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
=======
def monte_carlo_prediction(h: np.ndarray, gamma: float = 1, print_bool: bool = False) -> np.ndarray:
    """
    - function is doing first-visit policy evaluation
    - policy is implicitly given by history h

    INPUTS:
    h - A list of named tuples episodes
            - single episode has a form: episode(state=[0, 1, 2], action=[1, 2, 3], reward=[0, 0, 1])
    gamma - discount factor, float from [0, 1]

    OUTPUTS:
    q - numpy array with (16, 4) shape representing estimated state-action value function using
        Monte Carlo first-visit policy evaluation method and sampled interactions collected in h
    """
    q = np.zeros((16, 4), dtype=float)
    for n_episode, curr_episode in enumerate(h, start=1):
        gain = 0
        q_curr = np.zeros((16, 4))
        for step in reversed(range(len(curr_episode.states))):  # start from the last episode
            s, a, r = curr_episode.states[step], curr_episode.actions[step], curr_episode.rewards[step]
            gain = r + gamma * gain
            q_curr[s, a] = gain
            if print_bool and q_curr[s, a] != 0:
                print(f'({s}, {arrows[a]}, {q_curr[s, a]})')
        q += (q_curr - q) / n_episode
    return q


def monte_carlo_control(N_improvements: int = 10, epsilon: float = 0.5, gamma: float = 0.9):
    """
    INPUTS:
    epsilon: parameter in soft-epsilon Monte Carlo policy improvement method
             - when epsilon == 0 we use greedy update
    N_improvements: number of policy improvements
    gamma: discount factor used to calculate gain of a state-action pair

    OUTPUT:
    policy: improved policy
    """
    q = np.zeros((16, 4))
    pi = .25 * np.ones((4, 16))

    for i in range(N_improvements):
        # 1. sample an episode {s_0, a_0, r_0, ..., s_n, a_n, r_n} following the current policy
        history = sample_episodes(pi=pi, N_episodes=1)

        # 2. Monte Carlo Prediction stage:
        #            estimate state-action value function using sampled episode
        q = monte_carlo_prediction(h=history, gamma=gamma, print_bool=True)

        # 3. Monte Carlo Control stage:
        #            update policy using epsilon-greedy policy improvement method
        #            (for each state s: action with the highest q-value will have probability 1-epsilon
        #            and other actions will have equal probabilities that sums up to epsilon)
        policy = epsolon_greedy_policy_improvement(q_value=q, epsilon=epsilon)
        epsilon /= .9
    return q, policy


if __name__ == '__main__':
    mc_q_value, mc_policy = monte_carlo_control(N_improvements=1000, epsilon=0.2, gamma=0.9)
    #read_policy(pi=mc_policy)
    read_q_value(q=mc_q_value)
>>>>>>> f7069eda8304abfc62a611cbf550760d9009a56f
