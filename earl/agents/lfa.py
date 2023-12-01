import wandb
import numpy as np

from agents.utils.ea import EA

# Linear Function Approximation
class LFA(EA):
    def __init__(self, env, grid_size, num_obstacles):
        super(LFA, self).__init__(env, grid_size, num_obstacles)
        
        self.weights = None
        self.num_features = (2*self.state_dims) + self.action_dims
        
        self.alpha = 0.003
        self.epsilon_start = 1
        self.epsilon_decay = 0.99
                        
    def _init_wandb(self, problem_instance):
        config = super()._init_wandb(problem_instance)
        config.alpha = self.alpha
        config.epsilon = self.epsilon_start
        config.action_cost = self.action_cost
        config.num_episodes = self.num_episodes
        config.epsilon_decay = self.epsilon_decay
        
    def _decay_epsilon(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, 0.1)
        
    def _get_max_q_value(self, state):
        best_action = None
        max_q_value = -np.inf
        
        for action in range(self.action_dims):
            features = self._extract_features(state, action)
            q_value = np.dot(self.weights, features)
            
            if q_value > max_q_value:
                best_action = action
                max_q_value = q_value
        
        return best_action, max_q_value
        
    def _select_action(self, state):
        if self.rng.random() < self.epsilon:
            action = self.rng.integers(self.action_dims)
        else:
            action, _ = self._get_max_q_value(state)
        return action
    
    # Feature vector is a one hot encoding of the state and the action
    def _extract_features(self, state, action):
        state_features = np.zeros(self.state_dims*2)
        
        for i, cell in enumerate(state):
            if cell == 1:
                state_features[2*i] = 1
                state_features[2*i+1] = 0
            else:
                state_features[2*i] = 0
                state_features[2*i+1] = 1
                
        action_features = np.zeros(self.action_dims)
        action_features[action] = 1

        return np.concatenate([state_features, action_features])
    
    def _update_weights(self, state, action, reward, next_state, done):
        current_features = self._extract_features(state, action)
        current_q_value = np.dot(self.weights, current_features)
        
        _, max_next_q_value = self._get_max_q_value(next_state)
                
        td_target = reward + (1 - done) * max_next_q_value
        td_error = td_target - current_q_value
        
        self.weights = self.weights + self.alpha * td_error * current_features
    
    def _train(self, problem_instance, start_state):
        rewards = []
        best_actions = None
        best_reward = -np.inf
        
        for _ in range(self.num_episodes):
            done = False
            num_action = 0
            action_seq = []
            state = start_state
            while not done:
                num_action += 1
                action = self._select_action(state)
                reward, next_state, done = self._step(problem_instance, state, action, num_action)
                
                self._update_weights(state, action, reward, next_state, done)
                state = next_state
                action_seq += [action]
                
            self._decay_epsilon()
            
            rewards.append(reward)
            avg_rewards = np.mean(rewards[self.sma_window:])
            wandb.log({"Average Reward": avg_rewards})
            
            if reward > best_reward:
                best_reward = reward
                best_actions = action_seq
            
        return best_actions, best_reward
                
    def _generate_adaptations(self, problem_instance):
        self._init_wandb(problem_instance) 
        
        self.epsilon = self.epsilon_start
        self.weights = np.zeros(self.num_features)
                
        start_state = np.zeros(self.state_dims)
        best_actions, best_reward = self._train(problem_instance, start_state)
        
        wandb.log({'Final Reward': best_reward})
        wandb.log({'Final Actions': best_actions})
        wandb.finish()
        
        return best_actions