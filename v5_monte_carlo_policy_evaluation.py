"""
- dynamic programming is used to solve problems where the underlying model of the environment (transition probabilities) is known
    beforehand (or more precisely, model-based learning)
- here we don`t have knowledge of transition probabilities
- in Monte Carlo RL we have two steps:
    1. policy evaluation (finding the value function for a given random policy)
    2. and policy improvement step (finding the optimum policy)

Every visit Monte Carlo Policy Evaluation ALGORITHM:
1. initialize the policy and state-action value function
2. generate n episodes following current policy
   and save history of that episode {s0, a0, r0, s1, a1, r1,...}
3. calculate state-action value function using every visit monte carlo
   over the generated episodes
   - q(s, a) = average discounted gain from state s following the action a
4.
"""
import numpy as np
import random
import gym


def interact_with_environment(pi: np.ndarray, n_ep: int = 1, start: int = 0) -> list:
    '''
    function simulates game for n_ep times following the given policy pi

    INPUTS
    pi: (4, 16) numpy array
        - pi(a, s) is probability of agent taking action a from state s
    n_ep: number of episodes

    OPUTPUTS:
    h - list that collects all trajectories {s0, a0, r0, ...} as numpy arrays
      -> single trajectory is (3, n_steps) array where in
         first row are collected states, in second row actions and in third rewards.
    '''

    h = []
    for e in range(n_ep):
        env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
        T = np.zeros((3, 100), dtype=int)
        terminated, truncated = False, False
        curr_step = 0
        s = env.reset()[start]  # initial state is 0
        while not terminated and not truncated:
            # choose greedy action from 0=left, 1=down, 2=right, 3=up
            best_actions = (np.where(pi[:, s] == np.amax(pi[:, s])))[0]
            a = random.choice(best_actions)

            # take the action and save results
            T[0, curr_step] = s
            s, r, terminated, truncated, info = env.step(a)
            T[1, curr_step], T[2, curr_step] = a, r
            curr_step += 1
        T = T[:, :curr_step]
        h.append(T)
        env.close()
    return h


def monte_carlo_policy_evaluation(h: np.ndarray, gamma: float = 0.9) -> np.ndarray:
    """
    - function is used for policy evaluation
    - policy is implicitly given by history h

    INPUTS:
    h - list that collects all trajectories {s0, a0, r0, ...} as numpy arrays
        -> single trajectory is (3, n_steps) array where in
          first row are collected states, in second row actions and in third rewards.
    gamma - discount factor, float from [0, 1]

    OUTPUTS:
    q - numpy array with (16, 4) shape representing estimated state-action value function using
        Monte Carlo policy evaluation method
    """
    G = np.zeros((16, 4), dtype=float)  # average gain across all episodes for current policy
    for episode, H in enumerate(h, start=1):
        n_steps = H.shape[1]
        discount_factor = np.ones((1, n_steps))
        for i in range(1, n_steps):
            discount_factor[0, i] = discount_factor[0, i - 1] * gamma
        G_curr = np.zeros_like(G)  # calculates gain in current episode
        o_curr = np.zeros_like(G_curr)  # Number of occurrence of pair (state, action) in current episode
        for step in range(n_steps):
            state, action = H[0, step], H[1, step]
            o_curr[state][action] += 1
            G_curr[state][action] += np.sum(discount_factor[0, 0:(n_steps - step)] * H[2, step:])  # discounted
        G_curr[G_curr != 0] /= o_curr[G_curr != 0]  # find average value for G(state, action)
                                                    # for (state, action) pairs that occurred once or more
        if episode > 0:
            G = ((episode - 1) * G + G_curr) / episode  # gain of (state, action) pair is average across gains of that pair
                                                        # across all episodes (using incremental mean)
            """
            # here we can use running mean, as well
            alpha = 0.1
            G = G + alpha * (G - G_curr)
            """
    q = G
    return q


if __name__ == "__main__":
    """
    # test example for interact_with_environment
    n_episodes = 100
    policy = .25 * np.ones((4, 16))  # initial policy is random policy
    
    # make history 😅
    history = interact_with_environment(pi=policy, n_ep=n_episodes)
    print(history[0], '\n')
    print(history)
    """


    """
    # test example for monte_carlo_policy_evaluation:
    H1 = np.array([[0, 1, 1, 1], [2, 3, 3, 3], [0, 1, 2, 3]])
    H2 = np.array([[0, 1, 1, 1], [2, 3, 3, 3], [0, 1, 2, 3]])
    history = [H1]
    q = monte_carlo_policy_evaluation(h=history, gamma=0.9)
    print(q)
    
    result should be zero matrix with only following non-zero elements:
    q[0, 2] = 4.707
    q[1, 3] = 4.31
    """


