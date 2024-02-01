import gym
import numpy as np


def policy_evaluation(probs: dict, policy: np.ndarray,
                      gamma: float = .9, threshold: float = 1e-8,
                      v: np.ndarray = np.zeros((16,))):
    delta = threshold

    while delta >= threshold:
        delta = 0
        # update value function v for states 0-15
        for state in range(15):
            v_old = v[state]
            curr_sum = 0
            for action in range(4):
                for prob, next_state, reward, _ in env_probs[state][action]:
                    curr_sum += + prob * (reward + gamma * v[next_state])
                v[state] = policy[action][state] * curr_sum
            delta = max(delta, abs(v_old - v[state]))
    return v


env = gym.make('FrozenLake-v1', is_slippery=True)
env.reset()

env_probs = env.P  # dynamics of the system given as a dictionary where keys are states
                   # and values are dictionaries with actions as keys and following tuples as values:
                   # (p(curr_state | prev_action), state, reward, done)

# first example
uniform_policy = .25 * np.ones((4, 16))  # policy(action, state) - probability of taking action if we are in state
value_function_uniform = policy_evaluation(env_probs, uniform_policy)
print(value_function_uniform.reshape((4, 4)), '\n')

# second example
temp = np.array([[.1, .3, .5, .1]])
second_policy = temp.T * np.ones((4, 16))
value_function_second = policy_evaluation(env_probs, second_policy)
print(value_function_second.reshape((4, 4)), '\n')

env.close()


