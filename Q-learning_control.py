'''
more: https://towardsdatascience.com/introduction-to-reinforcement-learning-temporal-difference-sarsa-q-learning-e8f22669c366

One of the early breakthroughs in RL was the development of an
off-policy TD control algorithm known as Q-learning (Watkins, 1989) given bellow:

q(s_t, a_t) = q(s_t, a_t) +
              + alpha * (r_t + gamma * max_a(q(s_t+1, a)) - q(s_t, a_t))

- in this case q approximates optimal q-function, independent of current policy
- required for correct convergence is that all pairs continue to be updated


ALGORITHM for Q-learning (off-policy TD control)

algorithm parameter: step size alpha from (0, 1], small epsilon > 0
initialize q(s, a),
           - for all states s and all actions a arbitrarily
             except that q(terminal, _) = 0
loop for each episode:
    initialize state s
    loop for each step of episode:
        choose action a from s derived from q using epsilon-greedy method
        take action a, observe reward r and next state new_s
        q(s, a) = q(s, a) +
                  alpha * (r + gamma * max_a(q(new_s, new_a)) - q(s, a))
        s = new_s
    until S is terminal
'''