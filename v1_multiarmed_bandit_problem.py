import pandas as pd
import random
import numpy as np


# load dataset "clanci"
path = r'D:\Faks\MASTER\PyTorch\vežbe\data\clanci.csv'
df = pd.read_csv(path)


n_iter = 10000
# random selection
s = 0  # sum of rewards
for i in range(n_iter):
    a = random.randint(0, 9)  # curr action
    r = df.iloc[i, a]  # curr reward
    s += r
print(f'Broj nagrada koristeci random selection: {s}')

# upper confidence bound
t = 0
r = np.zeros((10,))
q = np.zeros((10,))  #  koliko puta su kliknuli taj clanak / koliko puta je taj clanak izabran da bude prvi
curr_upper_bound = np.zeros((10,))
n = np.zeros((10,))  # number of previous wins of selected action

for i in range(n_iter):
    if np.any(n == 0):
        a = np.where(n == 0)[0]
    else:
        curr_upper_bound = q + np.sqrt((2*np.log(t) / n))
        a = np.argmax(curr_upper_bound)
    r_curr = df.iloc[i, a]
    r[a] += r_curr
    n[a] += 1
    q[a] = r[a] / n[a]
    t += 1
print(f'Broj nagrada koristeci metod upper confidence bound: {np.sum(r)}')


# greedy action selection
q = np.zeros((10,))
n = np.zeros((10,))
s = 0
for i in range(n_iter):
    a = np.argmax(q)
    # print(a)
    n[a] += 1
    r_curr = df.iloc[i, a]
    s += r_curr
    q[a] += (r_curr - q[a]) / n[a]
print(f'Broj nagrada koristeci greedy method: {s}')


# epsilon greedy action selection, epsilon = prob of exploration
def epsilon_greedy(eps=.5):
    q = np.zeros((10,))
    n = np.zeros((10,))
    s = 0
    for i in range(n_iter):
        explore = np.random.binomial(1, eps)
        a_max = np.argmax(q)
        if explore == 0:  # exploit
            a = a_max
        else:  # explore
            a = random.randint(0, 8)
            if a == a_max:
                a = a % 9
        n[a] += 1
        r_curr = df.iloc[i, a]
        s += r_curr
        q[a] += (r_curr - q[a]) / n[a]
    return s

print(f'Broj nagrada koristeci epsilon=0.3 greedy method: {epsilon_greedy(eps=.3)}')
print(f'Broj nagrada koristeci epsilon=0.0 greedy method: {epsilon_greedy(eps=0.0)}')
print(f'Broj nagrada koristeci epsilon=1 greedy method: {epsilon_greedy(eps=1.0)}')
print(f'Broj nagrada koristeci epsilon=0.5 greedy method: {epsilon_greedy(eps=.5)}')



