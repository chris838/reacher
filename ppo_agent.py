import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions

import numpy as np

from rollout import Rollout
from policy_net import PolicyNet
from value_net import ValueNet

# Useful function from Shangton Zhang's awesome repo
# https://github.com/ShangtongZhang/DeepRL
def random_sample(indices, batch_size):
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]

class PPOAgent():

    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99, clipping_epsilon=0.1,
                 ppo_epochs=10, minibatch_size=64, rollout_length=1000, gae_lambda=0.95):
        self.lr = lr
        self.clipping_epsilon = clipping_epsilon
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.rollout_length = rollout_length

        self.policy = PolicyNet(state_size, action_size)
        self.value_estimator = ValueNet(state_size)
        self.rollout = Rollout(gamma=gamma, gae_lambda=gae_lambda)

    def start_episode(self):
        self.episode_rewards = []
        self.rollout.start_rollout()

    def act(self, state):

        # Check if the rollout is full and needs processing
        if len(self.rollout) == self.rollout_length:
            self.learn()
            self.rollout.start_rollout()

        # Derive action distribution from policy network
        means, sigmas = self.policy(state)
        action_distribution = torch.distributions.Normal(means, sigmas)
        action = action_distribution.sample()
        action_log_prob = action_distribution.log_prob(action)

        # Derive state value estimate from value network
        state_value = self.value_estimator(state).squeeze()

        # Record decision and return sampled action
        self.rollout.record_decision(state, state_value, action, action_log_prob)
        return action

    def finish_episode(self):
        self.learn()

    def record_outcome(self, reward):
        self.episode_rewards.append(reward)
        self.rollout.record_outcome(reward)

    def average_episode_return(self):
        return sum([r.mean().item() for r in self.episode_rewards])

    def get_current_policy_probs(self, states, actions):

        # For the given state/action pairs, create a distribution from the policy and get the log probs
        means, sigmas = self.policy(states)
        action_distribution = torch.distributions.Normal(means, sigmas)
        current_policy_log_probs = action_distribution.log_prob(actions)

        # Sum log probs over the possible actions
        current_policy_log_probs = current_policy_log_probs.sum(-1)

        return torch.exp(current_policy_log_probs)

    def learn(self):

        (states, actions, future_returns, normalised_advantages, original_policy_probs) = \
            self.rollout.flatten_trajectories()

        # Run through PPO epochs
        policy_optimiser = optim.Adam(self.policy.parameters(), lr=self.lr, eps=1e-5)
        value_estimator_optimiser = optim.Adam(self.value_estimator.parameters(), lr=self.lr, eps=1e-5)
        for ppo_epoch in range(self.ppo_epochs):

            # Sample the trajectories randomly in mini-batches
            for indices in random_sample(np.arange(states.shape[0]), self.minibatch_size):

                # Sample using sample indices
                states_sample = states[indices]
                actions_sample = actions[indices]
                future_returns_sample = future_returns[indices]
                normalised_advantages_sample = normalised_advantages[indices]
                original_policy_probs_sample = original_policy_probs[indices]

                # Use the current policy to get the probabilities for the sample states and actions
                # We use these to weight the likehoods, allowing resuse of the rollout
                current_policy_probs_sample = self.get_current_policy_probs(states_sample, actions_sample)

                # Define PPO surrogate and clip to get the policy loss
                sampling_ratio = current_policy_probs_sample / original_policy_probs_sample
                clipped_ratio = torch.clamp(sampling_ratio, 1 - self.clipping_epsilon, 1 + self.clipping_epsilon)
                clipped_surrogate = torch.min(
                    sampling_ratio * normalised_advantages_sample,
                    clipped_ratio * normalised_advantages_sample)
                policy_loss = -torch.mean(clipped_surrogate)

                # Define value estimator loss
                state_values_sample = self.value_estimator(states_sample).squeeze()
                value_estimator_loss = nn.MSELoss()(state_values_sample, future_returns_sample)

                # Update value estimator
                value_estimator_optimiser.zero_grad()
                value_estimator_loss.backward()
                nn.utils.clip_grad_norm_(self.value_estimator.parameters(), 0.75)
                value_estimator_optimiser.step()

                # Update policy
                policy_optimiser.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.75)
                policy_optimiser.step()
