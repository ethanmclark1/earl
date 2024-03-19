import torch
import numpy as np

from sklearn.preprocessing import OneHotEncoder

        
class RewardBuffer:
    def __init__(self, buffer_size, step_dims, num_actions, rng):       
        self.transition = torch.zeros(buffer_size, step_dims, dtype=torch.float) 
        self.reward = torch.zeros(buffer_size, dtype=torch.float)
        self.is_initialized = torch.zeros(buffer_size, dtype=torch.bool)
        
        self.encoder = OneHotEncoder(categories=[range(num_actions)], sparse_output=False)

        self.rng = rng
        # Managing buffer size and current position
        self.count = 0
        self.real_size = 0
        self.size = buffer_size
        
    def _increase_size(self):
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def add(self, state, action, reward, next_state):
        if isinstance(action, (int, np.integer)):
            action = np.array([action]).reshape(-1, 1)
            action = self.encoder.fit_transform(action).reshape(-1)
            
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        
        self.transition[self.count] = torch.cat((state, action, next_state), dim=0)
        self.reward[self.count] = torch.tensor((reward))
        self.is_initialized[self.count] = True

        self._increase_size()

    def sample(self, batch_size):
        initialized_idxs = torch.where(self.is_initialized == 1)[0]
        idxs = self.rng.choice(initialized_idxs, size=batch_size, replace=False)
        return idxs
    

class CommutativeRewardBuffer(RewardBuffer):
    def __init__(self, buffer_size, step_dims, action_dims, rng):
        super().__init__(buffer_size, step_dims, action_dims, rng)
        self.transition = torch.zeros((buffer_size, 2 ,step_dims), dtype=torch.float)
        self.reward = torch.zeros(buffer_size, 2, dtype=torch.float)
        
    def add(self, prev_state, action, prev_reward, commutative_state, prev_action, reward, next_state):
        if isinstance(action, (int, np.integer)):
            action = np.array([action]).reshape(-1, 1)
            action = self.encoder.fit_transform(action).reshape(-1)

            prev_action = np.array([prev_action]).reshape(-1, 1)
            prev_action = self.encoder.fit_transform(prev_action).reshape(-1)

        prev_state = torch.tensor(prev_state, dtype=torch.float)
        prev_action = torch.tensor(prev_action, dtype=torch.float)
        commutative_state = torch.tensor(commutative_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        
        step_0 = torch.cat((prev_state, action, commutative_state), dim=0)
        step_1 = torch.cat((commutative_state, prev_action, next_state), dim=0)
        self.transition[self.count] = torch.stack((step_0, step_1))
        self.reward[self.count] = torch.tensor((prev_reward, reward))
        self.is_initialized[self.count] = True

        self._increase_size()