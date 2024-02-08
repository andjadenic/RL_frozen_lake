from monte_carlo_control import monte_carlo_control
from sarsa import sarsa_control
from q_learning import q_learning
from utils import read_policy, read_q_value


q_value, policy = monte_carlo_control(N_improvements=1000, epsilon=0.2, gamma=0.9)

# policy, q_value = sarsa_control(N_improvements=100000, alpha=0.1, epsilon=0.1, gamma=0.9)

# policy, q_value = q_learning(N_improvements=10000, alpha=0.1, epsilon=0.1, gamma=0.9)


read_q_value(q_value)
read_policy(policy)