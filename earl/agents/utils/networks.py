import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


class RewardEstimator(nn.Module):
    def __init__(self, lr, step_size, gamma, dropout_rate, reward_range):
        super(RewardEstimator, self).__init__()
        input_dims = 3
        output_dims = 1
        self.fc1 = nn.Linear(in_features=input_dims, out_features=8)
        self.fc2 = nn.Linear(in_features=8, out_features=4)
        self.fc3 = nn.Linear(in_features=4, out_features=output_dims)
        
        self.dropout = nn.Dropout(p=dropout_rate)
        self.reward_range = reward_range
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=step_size, gamma=gamma)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc3(x)) * self.reward_range
        return x
    
    
# Permutation Invariant Neural Network 
# https://attentionneuron.github.io/
class PINN(nn.Module):
    def __init__(self, action_dims):
        super(PINN, self).__init__()
        self.query_size = 8
        self.message_size = 32
        self.action_dims = action_dims
        
        self.hx = None
        self.previous_action = torch.zeros(1, self.action_dims)
        
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
        self.previous_action = torch.zeros(1, self.action_dims)

    def forward(self, state):
        state = state.reshape(-1).unsqueeze(-1)
        state_dims = state.shape[0]

        if self.hx is None:
            self.hx = self.h0(state_dims)
            
        # Add previous action to the observation as the input for the LSTM
        x_pa = torch.cat([state, self.previous_action.repeat(state_dims, 1)], dim=-1)
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
        action_probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(action_probs, 1).item()
        self.previous_action = torch.eye(self.action_dims)[action]
        return action