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
    return pi, q


if __name__ == '__main__':

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
