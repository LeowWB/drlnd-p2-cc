## Report

### Learning Algorithm

The agent learns using the Deep Q-Learning algorithm, which uses a neural network to estimate the action-value function. Experience replay is used to allow the Q-network to revisit past experiences and learn from them multiple times; moving Q-targets are also used to reduce the likelihood of oscillation of the model weights.

The Q-network is a simple feedforward, fully-connected neural network with rectified linear unit activation. The neural network has one input layer of size 37 (corresponding to the dimensionality of the state space), two hidden layers of size 64 each, and an output layer of size 4 (corresponding to the dimensionality of the action space). The neural network thus maps states to a vector of q-values, one for each action.

The performance of the Q-network is determined by measuring the mean squared error loss between the output and the expected output. As each training example corresponds to a single (S,A,R,S') tuple, the error will only be considered with respect to one dimension of the output: the dimension that corresponds to the action A in the current tuple. Weights of the Q-network are then iteratively improved using an Adam optimizer.

**Hyperparameters**
* The size of the replay buffer is 10,000
* The size of a training minibatch is 64
* Gamma, the discount factor, is 0.99
* The learning rate is 0.0005
* The Q-network is trained with one minibatch after every 4 time steps
* Tau, the rate at which the target Q-network's weights are updated to match the local Q-network's, is 0.001

### Plot of Rewards

The below plot shows how the rewards received by the agent change as the number of episodes increases.

![Plot of rewards](./plot_of_rewards.png)

As can be seen from the above graph, the problem is solved by roughly the 600th episode.

### Ideas for Future Work

The code used in this notebook was adapted from the course materials with minimal changes. In fact, no major changes were needed in order to solve this problem. However, if the standard for solving was raised, hyperparameter selection using a Gaussian process might be a good way to improve performance. Prioritized experience replay, in which the tuples selected for the training minibatches are selected non-uniformly, might also be a good idea. This is because state transitions that involve collecting a banana are likely to be more important for learning than other state transitions.
