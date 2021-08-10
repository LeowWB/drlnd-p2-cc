## Report

### Learning Algorithm

This agent uses the DDPG (Deep Deterministic Policy Gradient) architecture, which is a subset of actor-critic methods. In this architecture, the actor deterministically approximates the optimal policy, mapping states to actions. Meanwhile the critic uses the actor's best believed action to approximate the optimal action value function, thus mapping (state, action) pairs to Q-values. 

DDPG is intuitively quite similar to DQN (Deep Q-Network) in terms of what it tries to do. Similar to the DQN, it also makes use of a replay buffer and fixed targets; unlike DQN, the targets in DDPG are updated via linear interpolation rather than direct copying.


**Hyperparameters**

### Plot of Rewards

The below plot shows how the rewards received by the agent change as the number of episodes increases.

As can be seen from the above graph, the problem is solved by roughly the ___th episode.

### Ideas for Future Work

The code used in this notebook was adapted from the course materials with relatively few changes. However, if the standard for solving was raised, hyperparameter selection (for both the actor and critic) using a Gaussian process might be a good way to improve performance. Prioritized experience replay, in which the tuples selected for the training minibatches are selected non-uniformly, might also be a good idea. This is because state transitions that involve moving into or out of the target region (and hence a direct change in the reward received) are likely to be more relevant than other state transitions. Finally, given the nature of the task, it may actually take a while for a new agent with no knowledge to receive its first non-zero reward. The lack of feedback that the agent would receive in the early phases might hinder learning; this can be partially alleviated using random restarts, to find a seed with which the agent quickly receives its first few rewards to kickstart the learning process.
