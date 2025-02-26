from utils import *
import copy


def policy_evaluation(P: dict, pi: np.ndarray, gamma: float = .9, theta: float = 1e-8):
    v = np.zeros((16,))

    # in-place policy evaluation
    max_diff = 1e10
    while max_diff > theta:
        max_diff = 0
        for s in range(15):
            vs = 0
            for a in range(4):
                for prob, s_new, r, _ in P[s][a]:
                    vs += pi[s, a] * prob * (r + gamma * v[s_new])

            max_diff = max(abs(v[s] - vs), max_diff)
            v[s] = vs
    return v


def q_value(P: dict, v: np.ndarray, s: int, gamma: float = 1):
    '''
    returns q values for each action for given state s
    '''
    q = np.zeros((4, ))
    for a in range(4):
        for prob, s_new, r, _ in P[s][a]:
            q[a] += prob * (r + gamma * v[s_new])
    return q


def policy_iteration(P: dict, pi: np.ndarray = 0.25 * np.ones((16, 4)),
                     theta: float = 1e-8, gamma: float = 0.9):
    converge = False
    q = np.zeros((16, 4))
    while not converge:
        # 1. policy evaluation
        v = policy_evaluation(P, pi, gamma=gamma, theta=theta)

        # 2. policy improvement
        pi_prim = np.zeros((16, 4))
        for s in range(15):
            q[s] = q_value(P, v, s, gamma)
            max_el = np.max(q[s])
            greedy_actions = []
            for a in range(4):
                if q[s][a] == max_el:
                    greedy_actions.append(a)
            pi_prim[s, greedy_actions] = 1 / len(greedy_actions)

        # 3. stop if pi converged
        if np.max(abs(policy_evaluation(P, pi)[0] - policy_evaluation(P, pi_prim)[0])) < theta * 1e2:
            converge = True

        # 4. Replace policy with new policy
        pi = copy.copy(pi_prim)

    pi = policy2greedy(pi)
    return pi, v, q


if __name__ == '__main__':

    #  Non-slipery Frozen Lake
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
    P = env.P  # transition probabilities
    env.close()

    pi_dp, v_dp, q_dp = policy_iteration(P=P, theta=1e-8, gamma=0.9)

    #plot_result(pi_dp, 'Non-slipery Frozen Lake: DP policy')
    #plot_result(q_dp, 'Non-slipery Frozen Lake: DP q value')

    mean_return = calculate_mean_return(pi=pi_dp, N_runs=50000, gamma=.9, is_slippery_bool=False)
    print(f'Non-slipery Frozen Lake:  DP {mean_return=}')


    #  Slipery Frozen Lake
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    P = env.P
    env.close()

    pi_dp, v_dp, q_dp = policy_iteration(P=P, theta=1e-8, gamma=0.9)

    # plot_result(pi_dp, 'Slipery Frozen Lake: DP policy')
    # plot_result(q_dp, 'Slipery Frozen Lake: DP q value')

    mean_return = calculate_mean_return(pi=pi_dp, N_runs=50000, gamma=.9, is_slippery_bool=True)
    print(f'Slipery Frozen Lake:  DP {mean_return=}')