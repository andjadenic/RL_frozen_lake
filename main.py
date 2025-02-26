<<<<<<< HEAD
import gymnasium as gym


arrows = [u'\u2190', u'\u2193', u'\u2192', u'\u2191']  # 0-left, 1-down, 2-right, 3-up
costum_map = ["SFFF",
              "FFFF",
              "FHFF",
              "FGFG"]

env = gym.make('FrozenLake-v1', desc=costum_map, map_name="4x4", is_slippery=True, render_mode='human')

state, info = env.reset(seed=42)
for episode in range(1, 3):
    terminated, truncated = False, False
    step = 0
    while not truncated and not terminated:
        action = env.action_space.sample()
        new_state, reward, terminated, truncated, info = env.step(action)
        print(f'e{episode} s{step}  :  ({state}, {arrows[action]}, {reward}, {terminated}, {truncated}, {info})')
        state = new_state
    state, info = env.reset()
    print('RESET')
    step += 1

env.close()
=======
from monte_carlo_control import monte_carlo_control
from sarsa import sarsa_control
from q_learning import q_learning
from utils import read_policy, read_q_value


q_value, policy = monte_carlo_control(N_improvements=1000, epsilon=0.2, gamma=0.9)

# policy, q_value = sarsa_control(N_improvements=100000, alpha=0.1, epsilon=0.1, gamma=0.9)

# policy, q_value = q_learning(N_improvements=10000, alpha=0.1, epsilon=0.1, gamma=0.9)


read_q_value(q_value)
read_policy(policy)
>>>>>>> f7069eda8304abfc62a611cbf550760d9009a56f
