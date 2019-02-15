import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions

class ValueNet(nn.Module):

    def __init__(self, state_size):
        super(ValueNet, self).__init__()

        # Hidden layers
        hidden_size = 32
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Output layer - single state value
        self.fc3= nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
