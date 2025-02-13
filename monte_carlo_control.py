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
import random
from utils import read_policy, read_q_value, epsolon_greedy_policy_improvement
import numpy as np
import gymnasium as gym
from collections import namedtuple


arrows = [u'\u2190', u'\u2193', u'\u2192', u'\u2191']  # 0-left, 1-down, 2-right, 3-up
episode = namedtuple('episode',
                     'states, actions, rewards, terminated, truncated')


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
        h.append(episode(states, actions, rewards, terminated, truncated))
    env.close()
    return h


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


def monte_carlo_control(N_improvements: int = 10, epsilon: float = 0, gamma: float = 0.9):
    """
    INPUTS:
    epsilon: parameter in soft-epsilon Monte Carlo policy improvement method
             - when epsilon == 0 we use greedy update
    N_improvements: number of policy improvements
    gamma: discount factor used to calculate gain of a state-action pair

    OUTPUT:
    policy: improved policy
    """
    q = np.ones((16, 4))
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
    return q, policy


if __name__ == '__main__':
    mc_q_value, mc_policy = monte_carlo_control(N_improvements=1000, epsilon=0.1, gamma=0.9)
    #read_policy(pi=mc_policy)
    read_q_value(q=mc_q_value)