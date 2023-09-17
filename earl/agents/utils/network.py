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