import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam


class DuelingDQN(nn.Module):
    def __init__(self, input_dims, output_dims, lr):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(input_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        
        self.adv_fc = nn.Linear(128, 128)
        self.adv_out = nn.Linear(128, output_dims)
        
        self.val_fc = nn.Linear(128, 64)
        self.val_out = nn.Linear(64, 1)
        
        self.optim = Adam(self.parameters(), lr=lr)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
                
        adv = F.relu(self.adv_fc(x))
        adv = self.adv_out(adv)
        
        val = F.relu(self.val_fc(x))
        val = self.val_out(val)
        
        return val + adv - adv.mean()
    
    
class MultiInputDuelingDQN(nn.Module):
    def __init__(self, state_dims, action_dims, output_dims, lr):
        super(MultiInputDuelingDQN, self).__init__()
        
        self.state_fc1 = nn.Linear(state_dims, 128)
        self.state_fc2 = nn.Linear(128, 128)
        
        self.action_fc1 = nn.Linear(action_dims, 128)
        self.action_fc2 = nn.Linear(128, 128)
        
        self.combined_fc = nn.Linear(128 + 128, 128)
        
        self.adv_fc = nn.Linear(128, 128)
        self.adv_out = nn.Linear(128, output_dims)
        
        self.val_fc = nn.Linear(128, 64)
        self.val_out = nn.Linear(64, 1)
        
        self.optim = Adam(self.parameters(), lr=lr)
    
    def forward(self, state, action):
        state_x = F.relu(self.state_fc1(state))
        state_x = F.relu(self.state_fc2(state_x))
        
        action_x = F.relu(self.action_fc1(action))
        action_x = F.relu(self.action_fc2(action_x))
        
        x = torch.cat((state_x, action_x), dim=-1)
        x = F.relu(self.combined_fc(x))
        
        adv = F.relu(self.adv_fc(x))
        adv = self.adv_out(adv)
        
        val = F.relu(self.val_fc(x))
        val = self.val_out(val)
        
        return val + adv - adv.mean()
