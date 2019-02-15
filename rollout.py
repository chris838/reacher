import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions

class Rollout():

    def __init__(self, gamma=0.99, gae_lambda=0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.start_rollout()

    def start_rollout(self):
        self.states = []
        self.state_values = []
        self.actions = []
        self.action_log_probs = []
        self.rewards = []

    def __len__(self):
        return len(self.states)

    def record_decision(self, state, state_value, action, action_log_prob):
        self.states.append(state)
        self.state_values.append(state_value)
        self.actions.append(action)
        self.action_log_probs.append(action_log_prob)

    def record_outcome(self, reward):
        self.rewards.append(reward)

    def flatten_trajectories(self):

        # Create tensors from states and actions
        states_tensors = torch.stack(self.states)
        actions_tensors = torch.stack(self.actions)

        # Calculate future return (at each step, for each trajectory)
        future_returns = [None] * len(self.rewards)
        future_return = torch.zeros_like(self.rewards[0])
        for i in reversed(range(len(self.rewards))):
            future_return *= self.gamma
            future_return += self.rewards[i]
            future_returns[i] = future_return.clone()
        future_returns = torch.stack(future_returns).detach()

        # Calculate TD errors
        td_errors = []
        state_values = self.state_values + [0] # terminal state has value 0
        for i in range(len(self.rewards)):
            td_error = self.rewards[i] + (self.gamma * state_values[i+1]) - state_values[i]
            td_errors.append(td_error)

        # Calculate advantages
        advantages = [None] * len(self.rewards)
        advantage = torch.zeros_like(self.rewards[0])
        for i in reversed(range(len(self.rewards))):
            advantage *= self.gae_lambda * self.gamma
            advantage += td_errors[i]
            advantages[i] = advantage.clone()
        advantages = torch.stack(advantages).detach()

        # Normalise advantages (across steps and trajectories)
        normalised_advantages = (advantages - advantages.mean()) / advantages.std()

        # Sum the log probabilities over the possible actions (at each step, for each trajectory)
        # We don't differentiate with respect to these, hence we detach them from the computation graph
        original_policy_log_probs = torch.stack(self.action_log_probs).sum(-1).detach()
        original_policy_probs = torch.exp(original_policy_log_probs)

        # Flatten trajectories
        return (states_tensors.view(-1, states_tensors.shape[-1]),
                actions_tensors.view(-1, actions_tensors.shape[-1]),
                future_returns.view(-1),
                normalised_advantages.view(-1),
                original_policy_probs.view(-1))
