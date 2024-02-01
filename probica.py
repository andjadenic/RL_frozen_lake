import numpy as np
import random


s = 2
q_value = np.array([[1, 1, 2, 3],
                    [.6, .7, .8, .19],
                    [10, 10, 10, 7]])

pi = np.ndarray((2, 4))
print(q_value[s, :])
print(q_value[s, :].shape, '\n')

best_actions_mask = np.where(q_value[s, :] == np.argmax(q_value[s, :]))
print(best_actions_mask)
