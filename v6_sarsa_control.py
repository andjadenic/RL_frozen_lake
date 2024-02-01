'''
more: https://towardsdatascience.com/introduction-to-reinforcement-learning-temporal-difference-sarsa-q-learning-e8f22669c366

SARSA is an on-policy TD method to solve the Control problem,
which means it tries to find an optimal policy for a given task.
- in on-policy learning, the optimal value function is learned
  from actions taken using the current policy 𝜋(𝑎|𝑠)
- we leverage the current policy to update Q(s, a)


SARSA CONTROL ALGORITHM:

algorithm parameter: step size alpha from (0, 1], small epsilon > 0
initialize q(s, a),
           - for all states s and all actions a arbitrarily
             except that q(terminal, _) = 0
loop for each episode:
    initialize state s
    choose action a from s derived from q using epsilon-greedy method
    loop for each step of episode:
        take action a
        take action a, observe reward r and next state new_s
        choose following action new_a from new_s using policy derived from q
                and epsilon-greedy method
        q(s, a) = q(s, a) + alpha * (r + gamma * q(new_s, new_a) - q(s, a))
        s = new_s
        a = new_a
    until S is terminal

pi(a, s) = argmax Q(a, s) for each state s
return pi
'''
import numpy as np
import gym
import random
from v5_monte_carlo_policy_evaluation import monte_carlo_policy_evaluation, interact_with_environment


def sarsa_control(epsilon: float = 0, discount: float = 0.9,
                  n_episodes: int = 100,
                  arbitrary_starting_state: bool = False):
    """
    UPDATE-UJ
    function for Sarsa on-policy one-step TD control:

    INPUTS:
    epsilon: parameter in soft-epsilon Monte Carlo policy improvement method
             - when epsilon == 0 we use greedy update
    n_improvements: number of policy evaluation-policy improvement pairs we do
    gamma: discount factor used to calculate gain od a state
    arbitrary_starting_state: weather starting state is arbitrary
                              - when set to False, starting state is 0
    arbitrary_policy: weather policy is arbatrary
                      - when set to False, policy is radom (uniform)

    OUTPUT:
    policy: improved policy
            - for update is used soft-epsilon (or greedy when epsilon=0) Monte Carlo method
    """

    # initialize Q
    Q = np.zeros((16, 4))

    for i in range(n_episodes):
        # initialize state
        s = 0
        if arbitrary_starting_state:
            s = random.randint(0, 15)

        env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
        terminated, truncated = False, False
        curr_step = 0
        s = env.reset()[s]  # initial state is s

        # choose action a from state s using epsilon-greedy policy derived from Q
        a = choose_epsilon_greedy_action(state=s, q_value=Q, e=epsilon)

        while not terminated and not truncated:

            # take the action and save results
            T[0, curr_step] = s
            s, r, terminated, truncated, info = env.step(a)
            T[1, curr_step], T[2, curr_step] = a, r
            curr_step += 1
        T = T[:, :curr_step]
        h.append(T)
        env.close()

        # policy evaluation: estimate state-action value function that following current policy
        history = interact_with_environment(pi=policy, n_ep=n_episodes, start=s)
        q_value = monte_carlo_policy_evaluation(h=history, gamma=discount)

        # policy improvement
        policy = monte_carlo_policy_improvement(epsilon=epsilon, q_value=q_value)

    return policy


def choose_epsilon_greedy_action(state: int, q_value: np.ndarray, e: float):
    pi = np.ndarray((2, 4))
    q_value[state, :]