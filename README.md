# Introduction to Reinforcement Learning using Gymnasium Frozen Lake

![Frozen Lake](https://gymnasium.farama.org/_images/frozen_lake.gif)

> Karpathy [said](https://www.youtube.com/watch?v=cdiD-9MMpb0) that learning RL agents from scratch is inefficient. It is okay to use RL if the agent has previous knowledge.

To give a gentle introduction to the field of reinforcement learning I wrote (this paper)[] that explains basic methods using the popular Toy Text environment in **Gymnasium Frozen Lake**.

###  Elements of Reinforcement Learning
Consult Chapter 1 in (paper)[] for elements of RL including state, action, reward, gain, value function, policy, and model.

### Mathematical Framework of Reinforcement Learning
Mathematical framework of Reinforcement Learning is **finite Markov Decision Process** discussed in Chapter 1 of the (paper)[].

### Bellman equation
* The Frozen Lake environment is used as an example to understand the Bellman equation better in (Chapter 3: Bellman Equations)[] 
* RL's most famous equation (a system of n equations with n unknown variables) is **Bellman equation**.
* ’Just’ solving that system of linear equations is computationally expensive with a time complexity of O(n
3). Because of that we find solutions numerically in (chapter 4: Dynamic Programming)[].

### Dynamic Programming
The set of algorithms for numerically estimating the optimal policy is called **Dynamic Programming**.
* DP algorithms use two processes **policy evaluation** and **policy improvement**. These two processes interact by leveraging **Generalized Policy Iteration (GPI)**, explained in detail in (chapter 4: Dynamic Programming)[].
* 

### Used literature
Reinforcement Learning: An Introduction by Richard S. Sutton and Andrew G. Barto

### ADD PDF
