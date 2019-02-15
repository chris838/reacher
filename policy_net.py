import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions

class PolicyNet(nn.Module):

    def __init__(self, state_size, action_size):
        super(PolicyNet, self).__init__()

        # Hidden layers
        hidden_size = 32
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Output layer of action means
        self.fc3= nn.Linear(hidden_size, action_size)

        # Standard deviations approximated seperately
        self.register_parameter('log_sigma', None)
        self.log_sigma = nn.Parameter(torch.zeros(action_size), requires_grad=True)

    def forward(self, x):

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        means = torch.tanh(self.fc3(x))
        sigmas = torch.exp(self.log_sigma).expand(means.shape)

        return means, sigmas
