import torch

class ReplayBuffer:
    def __init__(self, buffer_size, rng):       
        self.transition = torch.zeros(buffer_size, 3, dtype=torch.float) 
        self.reward = torch.zeros(buffer_size, dtype=torch.float)
        self.is_initialized = torch.zeros(buffer_size, dtype=torch.bool)

        self.rng = rng
        # Managing buffer size and current position
        self.count = 0
        self.real_size = 0
        self.size = buffer_size
        
    def _increase_size(self):
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def add(self, state, action, reward, next_state):        
        self.transition[self.count] = torch.tensor((state, action, next_state))
        self.reward[self.count] = torch.tensor((reward))
        self.is_initialized[self.count] = True

        self._increase_size()

    def sample(self, batch_size):
        initialized_idxs = torch.where(self.is_initialized == 1)[0]
        idxs = self.rng.choice(initialized_idxs, size=batch_size, replace=False)
        return idxs
    

class HallucinatedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, rng):
        super().__init__(buffer_size, rng)
        self.transition = torch.zeros((buffer_size,2,3), dtype=torch.float)
        self.reward = torch.zeros(buffer_size, 2, dtype=torch.float)
        
    def add(self, prev_state, prev_action, prev_reward, state, action, reward, next_state):
        self.transition[self.count] = torch.tensor(((prev_state, prev_action, state), (state, action, next_state)))   
        self.reward[self.count] = torch.tensor((prev_reward, reward))
        self.is_initialized[self.count] = True

        self._increase_size()