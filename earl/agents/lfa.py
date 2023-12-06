import math
import copy
import wandb
import random
import itertools
import numpy as np

from agents.utils.ea import EA

# Linear Function Approximation
class BasicLFA(EA):
    def __init__(self, env, grid_size, num_obstacles):
        super(BasicLFA, self).__init__(env, grid_size, num_obstacles)
        
        self.weights = None
        self.num_features = (2*self.state_dims) + self.action_dims
        
        self.alpha = 0.001
        self.epsilon_start = 1
        self.num_episodes = 5000
        self.epsilon_decay = 0.999
        self.sma_window = int(self.num_episodes * self.sma_percentage)

    def _init_wandb(self, problem_instance):
        config = super()._init_wandb(problem_instance)
        config.alpha = self.alpha
        config.epsilon = self.epsilon_start
        config.action_cost = self.action_cost
        config.num_episodes = self.num_episodes
        config.epsilon_decay = self.epsilon_decay
        config.action_success_rate = self.action_success_rate
        
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
        
        return td_error
    
    def _train(self, problem_instance):
        rewards = []
        
        for _ in range(self.num_episodes):
            done = False
            num_action = 0
            action_seq = []
            state = np.zeros(self.state_dims, dtype=int)
            while not done:
                num_action += 1
                action = self._select_action(state)
                reward, next_state, done = self._step(problem_instance, state, action, num_action)
                
                self._update_weights(state, action, reward, next_state, done)
                state = next_state
                action_seq += [action]
                
            self.epsilon *= self.epsilon_decay
            
            rewards.append(reward)
            avg_rewards = np.mean(rewards[-self.sma_window:])
            wandb.log({"Average Reward": avg_rewards})
            
    def _get_adaptation(self, problem_instance):
        done = False
        num_action = 0
        action_seq = []
        self.epsilon = 0
        state = np.zeros(self.state_dims, dtype=int)
        while not done:
            num_action += 1
            action = self._select_action(state)
            reward, next_state, done = self._step(problem_instance, state, action, num_action)
            state = next_state
            action_seq += [action]
        
        return action_seq, reward
            
    def _generate_adaptations(self, problem_instance):
        self._init_wandb(problem_instance) 
        
        self.epsilon = self.epsilon_start
        self.weights = np.zeros(self.num_features)
                
        self._train(problem_instance)
        adaptation, reward = self._get_adaptation(problem_instance)
        
        wandb.log({"Adaptation": adaptation})
        wandb.log({"Final Reward": reward})
        wandb.finish()
        
        return adaptation
    

class HallucinatedLFA(BasicLFA):
    def __init__(self, env, grid_size, num_obstacles):
        super(HallucinatedLFA, self).__init__(env, grid_size, num_obstacles)
        
        self.max_seq_len = 8
        
    def _sample_permutations(self, action_seq):
        permutations = {}
        permutations[tuple(action_seq)] = None
        
        tmp_action_seq = action_seq.copy()
        terminating_action = tmp_action_seq.pop()
        
        if len(tmp_action_seq) <= self.max_seq_len:
            permutations_no_term = itertools.permutations(tmp_action_seq)
            for permutation in permutations_no_term:
                permutations[tuple(permutation + (terminating_action,))] = None
        else:
            i = 1
            max_permutations = math.factorial(self.max_seq_len)
            while i < max_permutations:
                random.shuffle(tmp_action_seq)
                permutations[tuple(tmp_action_seq + [terminating_action])] = None
                i += 1   
        
        return list(permutations.keys())   
    
    # Hallucinate episodes by permuting the action sequence to simulate commutativity
    def _hallucinate(self, action_seq, reward):
        permutations = self._sample_permutations(action_seq)
        
        for permutation in permutations:
            state = np.zeros(self.state_dims, dtype=int)
            for action in permutation:
                next_state = copy.deepcopy(state)
                
                done = action == action_seq[-1]
                if done:
                    self._update_weights(state, action, reward, next_state, True)
                else:
                    # Transform action to fit it into the designated problem space
                    transformed_action = self._transform_action(action)
                    next_state[transformed_action] = 1
                    self._update_weights(state, action, 0, next_state, False)
                    state = next_state
    
    def _train(self, problem_instance):
        rewards = []
        
        for _ in range(self.num_episodes):
            done = False
            num_action = 0
            action_seq = []
            state = np.zeros(self.state_dims, dtype=int)
            while not done:
                num_action += 1
                action = self._select_action(state)
                reward, next_state, done = self._step(problem_instance, state, action, num_action)   
                             
                state = next_state
                action_seq += [action]
                
            self._hallucinate(action_seq, reward)
            self.epsilon *= self.epsilon_decay
            
            rewards.append(reward)
            avg_rewards = np.mean(rewards[-self.sma_window:])
            wandb.log({"Average Reward": avg_rewards}) 
            
    
class CommutativeLFA(BasicLFA):
    def __init__(self, env, grid_size, num_obstacles):
        super(CommutativeLFA, self).__init__(env, grid_size, num_obstacles)