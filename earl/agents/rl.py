class RL:
    def __init__(self, state_dims, action_dims):
        self.state_dims = state_dims
        self.action_dims = action_dims
    
    def select_action(self, state):
        raise NotImplementedError