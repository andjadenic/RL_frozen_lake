"""
MONTE CARLO CONTROL
~this file follows monte carlo policy evaluation~

- we know how to calculate state-action value function for any policy
  using monte carlo policy evaluation
- goal here is to calculate optimal policy
- in dynamic programing for finding optimal policy we used bellman's equation
  that uses transitional probabilities
- here, policy improvement is done by updating policy following:

    1. GREEDY POLICY UPDATE with respect to the current state-action value function

    or

    2. EPSILON-SOFT POLICY UPDATE with respect to the current state-action value function

- we don't need the model for these improvements
- for policy improvement we use knowledge about current policy in these Monte Carlo Reinforcement Learning methods.
  They are called ON POLICY METHODS
- there are also OFF POLICY METHODS studied next file


ALGORITHM for Monte Carlo control:

initialize:
    - starting state as pre-defined or arbitrary
    - policy as random or arbitrary
repeat:
1. estimate state-action value function that following current policy
2. update policy with greedy or epsilon-soft policy with respect to estimated state-action value function
until satisfied

"""
import numpy as np
import random
from v5_monte_carlo_policy_evaluation import monte_carlo_policy_evaluation, interact_with_environment


def monte_carlo_policy_improvement(q_value: np.ndarray, epsilon: float = 0):
    """
    function does single update on policy using q_value and
    soft-epsilon (greedy when epsilon == 0) Monte Carlo update

    INPUTS
    q_value: (4, 16) numpy array
            -> q_value[a, s] is probability of agent taking action a from state s
    epsilon: float number from [0, 1] determining how much policy is greedy
             -> probability for taking argmax action is epsilon/4 + 1 - epsilon
                and probability of taking other actions is epsilon/4
             -> when epsilon == 0, policy is greedy
             -> when epsilon == 1, policy is random (uniform)

    OUTPUT
    policy: improved policy
    """
    policy = np.zeros((4, 16))

    # policy improvement
    for state in range(16):
        best_actions = np.where(
            q_value[:][state] == np.max(q_value[:][state]))  # all greedy actions are in array best_action[0]
        if len(best_actions[0]) > 1:
            sample = random.randint(0, len(best_actions[0]) - 1)
            best_action = best_actions[0][sample]
        else:
            best_action = best_actions[0][0]
        policy[:, state] = epsilon / 4
        policy[best_action, state] += (1 - epsilon)  # best action is greedy action
    return policy


def monte_carlo_control(epsilon: float = 0, discount: float = 0.9,
                        n_episodes: int = 1, n_improvements: int = 100,
                        arbitrary_starting_state: bool = False, arbitrary_policy: bool = False):
    """
    ALGORITHM for Monte Carlo control:
    initialize:
        - starting state as pre-defined or arbitrary
        - policy as random or arbitrary
    for n_improvements times repeat:
    1. estimate state-action value function that following current policy
    2. update policy with greedy or epsilon-soft policy with respect to estimated state-action value function

    INPUTS:
    epsilon: parameter in soft-epsilon Monte Carlo policy improvement method
             - when epsilon == 0 we use greedy update
    n_episodes: for how many episodes we play the game using existing policy to evaluate the policy
                using Monte Carlo policy evaluation
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
    # initialize state
    s = 0
    if arbitrary_starting_state:
        s = random.randint(0, 15)

    # initialize policy
    policy = .25 * np.ones((4, 16))
    if arbitrary_policy:
        policy = np.random.rand((4, 16))  # random floats from [0, 1]
        policy /= policy.sum(axis=1, keepdims=True)  # normalize elements so that sum of each row is 1

    for i in range(n_improvements):
        # policy evaluation: estimate state-action value function that following current policy
        history = interact_with_environment(pi=policy, n_ep=n_episodes, start=s)
        q_value = monte_carlo_policy_evaluation(h=history, gamma=discount)

        # policy improvement
        policy = monte_carlo_policy_improvement(epsilon=epsilon, q_value=q_value)

    return policy


if __name__ == '__main__':
    '''
    # testing example for Monte Carlo control with single evaluation and improvement

    # initialize starting state
    s = 0
    if arbitrary_starting_state:
        s = random.randint(0, 15)

    # initialize policy
    policy = .25 * np.ones((4, 16))
    if arbitrary_policy:
        policy = np.random.rand((4, 16))  # random floats from [0, 1]
        policy /= policy.sum(axis=1, keepdims=True)  # normalize elements so that sum of each row is 1

    # policy evaluation: estimate state-action value function that following current policy
    history = interact_with_environment(pi=policy, n_ep=n_episodes)
    q_value = monte_carlo_policy_evaluation(h=history, gamma=discount)

    print(monte_carlo_policy_improvement(q_value=q_value))
    '''

    # testing Monte Carlo control for 100 iterations
    p = monte_carlo_control(n_improvements=100)
    print(p)


    # results for 10 policy inprovement iterations, each playing 1000 episodes
    '''
    [[0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 1. 0. 1. 0. 0. 0.]
     [0. 0. 1. 1. 0. 0. 0. 1. 0. 1. 0. 0. 0. 0. 0. 0.]
     [1. 1. 0. 0. 0. 1. 1. 0. 1. 0. 0. 1. 0. 1. 1. 1.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
     
     0 - desno
     1 - desno
     2
    '''