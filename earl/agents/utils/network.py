import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam


class BayesianLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BayesianLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # mu and rho for weight and bias
        # use rho instead of sigma because rho can take on any value in the real number line
        self.w_mu = nn.Parameter(torch.Tensor(output_dim, input_dim).normal_(0, 0.01))
        self.w_rho = nn.Parameter(torch.Tensor(output_dim, input_dim).normal_(0, 0.01))
        self.b_mu = nn.Parameter(torch.Tensor(output_dim).normal_(0, 0.01))
        self.b_rho = nn.Parameter(torch.Tensor(output_dim).normal_(0, 0.01))
        
    def forward(self, x):
        # generate Gaussian noise for weights and biases
        w_epsilon = torch.normal(0, 1, size=(self.output_dim, self.input_dim))
        b_epsilon = torch.normal(0, 1, size=(self.output_dim,))

        # softplus function converts rho to sigma while ensuring sigma is positive
        w_sigma = torch.log1p(torch.exp(self.w_rho))
        b_sigma = torch.log1p(torch.exp(self.b_rho))
        
        # reparameterization trick to get point estimate of weights and biases
        w = self.w_mu + w_sigma * w_epsilon
        b = self.b_mu + b_sigma * b_epsilon
        
        return F.linear(x, w, b)
    

# uncertainty is encoded in the network weights and biases
class BayesianDQN(nn.Module):
    def __init__(self, input_dims, output_dims, lr):
        super(BayesianDQN, self).__init__()
        self.fc1 = BayesianLinear(input_dims, 128)
        self.fc2 = BayesianLinear(128, 128)
        self.fc3 = BayesianLinear(128, output_dims)
        
        self.optim = Adam(self.parameters(), lr=lr)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    
class MultiInputBayesianDQN(nn.Module):
    def __init__(self, state_dims, action_dims, output_dims, lr):
        super(MultiInputBayesianDQN, self).__init__()
        
        self.state_fc1 = BayesianLinear(state_dims, 128)
        self.state_fc2 = BayesianLinear(128, 128)
        
        self.action_fc1 = BayesianLinear(action_dims, 128)
        self.action_fc2 = BayesianLinear(128, 128)
        
        self.combined_fc = BayesianLinear(128 + 128, output_dims)
        
        self.optim = Adam(self.parameters(), lr=lr)
    
    def forward(self, state, action):
        state_x = F.relu(self.state_fc1(state))
        state_x = F.relu(self.state_fc2(state_x))
        
        action_x = F.relu(self.action_fc1(action))
        action_x = F.relu(self.action_fc2(action_x))
        
        x = torch.cat((state_x, action_x), dim=-1)
        return self.combined_fc(x)