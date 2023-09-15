import numpy as np

class RL:
    def __init__(self, env):
        self._init_hyperparams()
        
        self.env = env
        state_dims = env.observation_space.n
        action_dims = env.action_space.n
        self.q_table = np.zeros((state_dims, action_dims))
        self.adaptive_q_table = np.zeros((state_dims, action_dims))
        
    def _init_hyperparams(self):
        self.alpha = 0.7
        self.epsilon = 1
        self.gamma = 0.95
        self.epsilon_decay = 0.9999
    
    def _get_action(self, state):
        if np.random.random() > self.epsilon:
            action = np.argmax(self.q_table[state,:])
        else:
            action = np.random.randint(self.action_dims)
        return action
    
    def _learn(self, transition):
        state, action, reward, next_state, _ = transition
        next_q = self.q_table[next_state]
        q_target = reward + self.gamma * np.max(next_q)
        
        td_error = q_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
            
    def train(self, num_epochs):        
        for epoch in range(num_epochs):
            step = 0
            done = False
            state = self.env.reset()[0]
            while not done:
                action = self._get_action(state)
                next_state, reward, truncation, termination, _ = self.env.step(action)
                done = truncation or termination
                self._learn((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward
                step += 1
            
            self.epsilon *= self.epsilon_decay
                    
    def evaluate(self, num_epochs):
        total_step = 0
        total_reward = 0
        
        for epoch in range(num_epochs):
            step = 0
            done = False
            state = env.reset()[0]
            while not done:
                action = np.argmax(self.q_table[state,:])
                next_state, reward, truncation, termination, _ = self.env.step(action)
                done = truncation or termination
                state = next_state
                step += 1
            
            total_step += step
            total_reward += reward
        
        avg_step = total_step / num_epochs
        success_rate = (total_reward / num_epochs) * 100
                    
        print(f'RL Average Episode Length: {avg_step}\nRL Success Rate: {success_rate}\n')