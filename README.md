# Introduction to Reinforcement Learning using Gymnasium Frozen Lake

![Frozen Lake](https://gymnasium.farama.org/_images/frozen_lake.gif)

> Karpathy [said](https://www.youtube.com/watch?v=cdiD-9MMpb0) that learning RL agents from scratch is inefficient. It is okay to use RL if the agent has previous knowledge.

To give a gentle introduction to the field of reinforcement learning I wrote (this paper)[] that explains basic methods using the popular Toy Text environment in **Gymnasium Frozen Lake**.

###  Elements of Reinforcement Learning
Consult Chapter 1 in (paper)[] for elements of RL including state, action, reward, gain, value function, policy, and model.

### Mathematical Framework of Reinforcement Learning
The mathematical framework of Reinforcement Learning is **finite Markov Decision Process** discussed in Chapter 1 of the (paper)[].

### Bellman equation
* The Frozen Lake environment is used as an example to understand the Bellman equation better in (Chapter 3: Bellman Equations)[] 
* RL's most famous equation (a system of n equations with n unknown variables) is **Bellman equation**.
* ’Just’ solving that system of linear equations is computationally expensive with a time complexity of O(n
3). Because of that we find solutions numerically in (chapter 4: Dynamic Programming)[].

### Dynamic Programming
The set of algorithms for numerically estimating the optimal policy is called **Dynamic Programming**.
* DP algorithms use two processes **policy evaluation** and **policy improvement**. These two processes interact by leveraging **Generalized Policy Iteration (GPI)**, explained in detail in (chapter 4: Dynamic Programming)[].
* File (dynamic_programming.py)[] uses GPI DP for estimating optimal policy in the frozen lake environment.

### Monte Carlo Mehod
* Monte Carlo method does not require knowledge of the dynamics of the environment (like DP).
* MC method uses just the definition of the state-action value function as the expected value and the basic
statistical method (Monte Carlo method) to estimate the state-action value function.
* Once we have an estimation of a state-value function, we can easily estimate optimal policy.
* Python code for Frozen Lake vith visualizations is in the file [monte_carlo_control.py](https://github.com/andjadenic/RL_frozen_lake/blob/main/monte_carlo_control.py)

### Temporal Difference
* The third set of algorithms for estimating the optimal policy that (just like MC) does not require knowledge of the dynamics of the environment are TD algorithms.
* Methods use estimates to estimate (they bootstrap) state-action value functions. They
combine DP and MC methods in that sense.
* There are a few versions of the TD algorithm for estimating the optimal policy. The most popular are used in the Frozen Lake environment:
  * **SARSA** (one-step TD)  [Python code in Frozen Lake](https://github.com/andjadenic/RL_frozen_lake/blob/main/sarsa.py)
  * **n step SARSA** (n-step TD) [Python code in Frozen Lake](https://github.com/andjadenic/RL_frozen_lake/blob/main/sarsa.py)
  * **expected SARSA** (one-step TD) [Python code in Frozen Lake](https://github.com/andjadenic/RL_frozen_lake/blob/main/sarsa.py)
  * **Q-learning** (one-step TD) [Python code in Frozen Lake](https://github.com/andjadenic/RL_frozen_lake/blob/main/q_learning.py)
 
### Algorithm comparison and visualizations
All mentioned algorithms are used in the Frozen Lake example. Results are visualized and compared in (notebook)[].


### Used literature
Reinforcement Learning: An Introduction by Richard S. Sutton and Andrew G. Barto
>>>>>>> f7069eda8304abfc62a611cbf550760d9009a56f
