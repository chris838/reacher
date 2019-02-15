# Udacity Deep Reinforcement Learning Nanodegree
# Project 2: Continuous Control

This project is an attempt to solve the reinforcement learning test environment called Reacher, which simulates 20 robotic arms in 3D and tasks the agent with controlling their movements in order to reach a specified target region. The required score of 30 (average return over 100 consecutive episodes) was achieved after 92 episodes, using a version of proximal policy optimisation.


## Learning algorithm

All arms are simultaneously controlled by a single agent using the same policy, effectively allowing us to train the agent on 20 independent environments in parallel. The policy is based on [proximal policy optimisation](https://arxiv.org/abs/1707.06347) (PPO). We augment this with a baseline, similar to the one described in [generalised advantage estimation](https://arxiv.org/abs/1506.02438).

The agent uses two separate neural networks, one for the policy and one for the action.

The policy network takes in a 33-dimensional state vector and outputs the 4 means and standard deviations which define probability density functions over the continuous action space (this approach was outlined in [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)).

The value network takes in a 33-dimensional state vector and outputs an estimate for the value of the state.

The agents actions are taken according to the policy and a rollout of states, state values, actions, action log probabilities (i.e. log probability of being selected under the policy) and rewards is collected. Once the rollout reaches sufficient length, a learning step is triggered.

During learning, we calculate returns and advantages from the rollout, then sample random mini-batches and use them to improve both networks. The rollout is re-used several times by weighting according to an importance sampling ratio, with clipping used to ensure the ratio stays reasonably close to 1.


# Neural network architecture

Both networks are distinct, yet share essentially the same architecture but with different outputs.

Below is the summary for the policy network. Note there are 4 additional trainable model parameters used to calculate the standard deviations, which aren't included here. These are independent of the input. On the hidden layers we use the `ReLU` activation function. On the outputs we use `tanh` to produce means in the desired range (-1 to 1) and `exp` to produce standard deviations that are positive.

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Linear-1                   [-1, 32]           1,088
                Linear-2                   [-1, 32]           1,056
                Linear-3                    [-1, 4]             132
    ================================================================
    Total params: 2,276
    Trainable params: 2,276
    Non-trainable params: 0
    ----------------------------------------------------------------

Below is the summary for the value network. Again we use `ReLU` on the hidden layers. On the output we use `sigmoid` to produce output in the range -1 to 1 (technically this is incorrect since there's no reason why values should be clamped to this range; fortunately it didn't seem to adversely affect the result).

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Linear-1                   [-1, 32]           1,088
                Linear-2                   [-1, 32]           1,056
                Linear-3                    [-1, 1]              33
    ================================================================
    Total params: 2,177
    Trainable params: 2,177
    Non-trainable params: 0
    ----------------------------------------------------------------


# Results

As required, the agent is able to receive an average reward (over 100 episodes, and over all 20 agents) of at least +30. This is reached after just 92 episodes.

![Reward Graph of PPO](https://github.com/chris838/reacher/blob/master/results/ppo-reacher.png)



# Future ideas for improvement

Training was extremely slow, anything which could improve this would make future experimentation easier. CUDA didn't seem to make any difference but it's possible this was a misconfiguration issue.

The GAE baseline used to augment the PPO policy estimate could be swapped out for a low variance method such as TD or n-step TD. This could then correctly be described as an actor-critic setup.


# Bugs and fixes

The sigmoid activation function on the output of the value network is incorrect and this should be fixed. Another minor bug, which didn't adversely effect training, is that the final state and state value is missing from the rollout.
