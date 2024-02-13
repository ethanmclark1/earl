import torch

torch.manual_seed(seed=42)

        
class RewardBuffer:
    def __init__(self, buffer_size, step_dims):       
        self.transition = torch.zeros(buffer_size, step_dims, dtype=torch.float) 
        self.reward = torch.zeros(buffer_size, dtype=torch.float)
        self.is_initialized = torch.zeros(buffer_size, dtype=torch.bool)

        # Managing buffer size and current position
        self.count = 0
        self.real_size = 0
        self.size = buffer_size
        
    def _increase_size(self):
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def add(self, s_1, b, s_prime, r_1):        
        self.transition[self.count] = torch.tensor((s_1, b, s_prime))
        self.reward[self.count] = torch.tensor((r_1))
        self.is_initialized[self.count] = True

        self._increase_size()

    def sample(self, batch_size):
        initialized_idxs = torch.where(self.is_initialized == 1)[0]
        random_indices = torch.randint(0, len(initialized_idxs), (batch_size,))
        idxs = initialized_idxs[random_indices]
        return idxs
    

class CommutativeRewardBuffer(RewardBuffer):
    def __init__(self, buffer_size, step_dims):
        super().__init__(buffer_size, step_dims)
        self.transition = torch.zeros((buffer_size,2,step_dims), dtype=torch.float)
        self.reward = torch.zeros(buffer_size, 2, dtype=torch.float)
        
    def add(self, s, b, s_2, a, s_prime, r_0, r_1):
        self.transition[self.count] = torch.tensor(((s, b, s_2), (s_2, a, s_prime)))   
        self.reward[self.count] = torch.tensor((r_0, r_1))
        self.is_initialized[self.count] = True

        self._increase_size()