import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam

# Uncertainty is encoded in the network weights and biases
class BayesianLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BayesianLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Mu and rho for weight and bias
        # Use rho instead of sigma because rho can take on any value in the real number line
        self.w_mu = nn.Parameter(torch.Tensor(output_dim, input_dim).normal_(0, 0.01))
        self.w_rho = nn.Parameter(torch.Tensor(output_dim, input_dim).normal_(0, 0.01))
        self.b_mu = nn.Parameter(torch.Tensor(output_dim).normal_(0, 0.01))
        self.b_rho = nn.Parameter(torch.Tensor(output_dim).normal_(0, 0.01))
        
    def forward(self, x):
        # Generate Gaussian noise for weights and biases
        w_epsilon = torch.normal(0, 1, size=(self.output_dim, self.input_dim))
        b_epsilon = torch.normal(0, 1, size=(self.output_dim,))

        # Softplus function converts rho to sigma while ensuring sigma is positive
        w_sigma = torch.log1p(torch.exp(self.w_rho))
        b_sigma = torch.log1p(torch.exp(self.b_rho))
        
        # Reparameterization trick to get point estimate of weights and biases
        w = self.w_mu + w_sigma * w_epsilon
        b = self.b_mu + b_sigma * b_epsilon
        
        return F.linear(x, w, b)
    
class BayesianDQN(nn.Module):
    def __init__(self, state_dims, action_dims, lr):
        super(BayesianDQN, self).__init__()
        self.fc1 = BayesianLinear(state_dims, 32)
        self.fc2 = BayesianLinear(32, 32)
        self.fc3 = BayesianLinear(32, action_dims)
        
        self.optim = Adam(self.parameters(), lr=lr)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    

# Permutation Invariant Neural Network 
# https://attentionneuron.github.io/
class PINN(nn.Module):
    def __init__(self, action_dims, temperature):
        super(PINN, self).__init__()
        self.query_size = 8
        self.message_size = 32
        self.temperature = temperature
        self.action_dims = action_dims
        
        self.hx = None
        self.previous_action = torch.zeros(self.action_dims)
        
        self.lstm = nn.LSTMCell(input_size=self.action_dims+1, hidden_size=self.query_size)
        self.q = torch.from_numpy(self.pos_table(16, self.query_size)).float()
        self.fq = nn.Linear(in_features=self.query_size, out_features=self.message_size, bias=False)
        self.fk = nn.Linear(in_features=self.query_size, out_features=self.message_size, bias=False)
        self.head = nn.Sequential(nn.Linear(in_features=16, out_features=action_dims))
    
    # Generate table of positional encodings
    def pos_table(self, n, dim):
        def get_angle(x, h):
            return x / np.power(10000, 2 * (h // 2) / dim)

        def get_angle_vec(x):
            return [get_angle(x, j) for j in range(dim)]

        table = np.array([get_angle_vec(i) for i in range(n)]).astype(float)
        table[:, 0::2] = np.sin(table[:, 0::2])
        table[:, 1::2] = np.cos(table[:, 1::2])
        return table
            
    def h0(self, state_dims):
        return (torch.zeros(state_dims, self.query_size), torch.zeros(state_dims, self.query_size))
    
    def reset(self):
        self.hx = None
        self.previous_action = torch.zeros(self.action_dims)

    def forward(self, state):
        state = state.unsqueeze(-1)
        state_dims = state.shape[0]

        if self.hx is None:
            self.hx = self.h0(state_dims)
            
        # Add previous action to the observation as the input for the LSTM
        x_pa = torch.cat([state, self.previous_action.repeat(self.action_dims-1, 1)], dim=-1)
        self.hx = self.lstm(x_pa, self.hx)
        
        # Compute attention matrix
        # Query: positional encoding of q
        q = self.fq(self.q)
        # Key: f_k(o_t[i], a_{t-1})
        # self.hx[0] is the hidden state of the LSTM
        k = self.fk(self.hx[0])
        dot = torch.matmul(q, k.T)
        attention_scores = torch.div(dot, math.sqrt(self.query_size))
        attention_weights = torch.tanh(attention_scores)
        # Value: f_v is a pass through function hence we can use the state as the value
        m = torch.tanh(torch.matmul(attention_weights, state))
        
        logits = self.head(m.T)
        scaled_logits = torch.div(logits, self.temperature)
        action_probs = F.softmax(scaled_logits, dim=-1)
        action = torch.multinomial(action_probs, 1).item()
        self.previous_action = torch.eye(self.action_dims)[action]
        return action
    

# Twin Delayed DDPG (TD3)
# https://spinningup.openai.com/en/latest/algorithms/td3.html
class Actor(nn.Module):
    def __init__(self, state_dims, action_dims, lr):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dims, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_dims)
        
        self.optim = Adam(self.parameters(), lr=lr)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
        
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, lr):
        super(Critic, self).__init__()

        # Q1 architecture
        self.fc1 = nn.Linear(state_dim + action_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(16, 1)

        # Q2 architecture
        self.fc4 = nn.Linear(state_dim + action_dim, 32)
        self.fc5 = nn.Linear(32, 32)
        self.fc6 = nn.Linear(32, 1)
        
        self.optim = Adam(self.parameters(), lr=lr)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        # Q1 architecture
        x1 = F.relu(self.fc1(xu))
        x1 = F.relu(self.fc2(x1))
        x1 = self.fc3(x1)

        # Q2 architecture
        x2 = F.relu(self.fc4(xu))
        x2 = F.relu(self.fc5(x2))
        x2 = self.fc6(x2)
        return x1, x2

    # More efficient to only compute Q1
    def get_Q1(self, state, action):
        xu = torch.cat([state, action], 1)
        x1 = F.relu(self.fc1(xu))
        x1 = F.relu(self.fc2(x1))
        x1 = self.fc3(x1)
        return x1