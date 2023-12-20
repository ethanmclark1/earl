import math
import wandb
import random
import itertools
import numpy as np

from agents.utils.ea import EA

# Linear Function Approximation
class BasicLFA(EA):
    def __init__(self, env, rng, percent_obstacles):
        super(BasicLFA, self).__init__(env, rng, percent_obstacles)
        
        self.weights = None
        # Add a dummy action (+1) to terminate the episode
        self.action_dims = env.observation_space.n + 1
        self.num_features = (2*self.state_dims) + self.action_dims
        
        self.alpha = 0.0002
        self.sma_window = 100
        self.epsilon_start = 1
        self.num_episodes = 7500
        self.epsilon_decay = 0.999

    def _init_wandb(self, problem_instance):
        config = super()._init_wandb(problem_instance)
        config.alpha = self.alpha
        config.max_action = self.max_action
        config.action_cost = self.action_cost
        config.num_episodes = self.num_episodes
        config.epsilon_decay = self.epsilon_decay
        config.percent_obstacles = self.percent_obstacles
        config.action_success_rate = self.action_success_rate
        config.configs_to_consider = self.configs_to_consider
        
    # Feature vector is a one hot encoding of the state and the action
    def _extract_features(self, state, action):
        state_features = np.zeros(self.state_dims*2)
        
        for i, row in enumerate(state):
            for j, cell in enumerate(row):
                index = (i * self.num_cols + j) * 2
                if cell == 1:
                    state_features[index] = 1
                    state_features[index + 1] = 0
                else:
                    state_features[index] = 0
                    state_features[index + 1] = 1
                
        action_features = np.zeros(self.action_dims)
        action_features[action] = 1

        return np.concatenate([state_features, action_features])
        
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
    
    def _update_weights(self, state, action, reward, next_state, done):
        current_features = self._extract_features(state, action)
        current_q_value = np.dot(self.weights, current_features)
        
        _, max_next_q_value = self._get_max_q_value(next_state)
                
        td_target = reward + (1 - done) * max_next_q_value
        td_error = td_target - current_q_value
        
        self.weights = self.weights + self.alpha * (current_features * td_error)
    
    def _train(self, problem_instance):
        rewards = []
        
        for _ in range(self.num_episodes):
            done = False
            num_action = 0
            episode_reward = 0
            state = self._generate_state(problem_instance)
            while not done:
                num_action += 1
                action = self._select_action(state)
                reward, next_state, done = self._step(problem_instance, state, action, num_action)
                
                self._update_weights(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                
            self.epsilon *= self.epsilon_decay
            
            rewards.append(episode_reward)
            avg_rewards = np.mean(rewards[-self.sma_window:])
            wandb.log({"Average Reward": avg_rewards})
    
    def _get_final_adaptations(self, problem_instance):
        done = False
        num_action = 0
        action_seq = []
        self.epsilon = 0
        state = np.zeros(self.grid_dims, dtype=int)
        while not done:
            num_action += 1
            action = self._select_action(state)
            _, next_state, done = self._step(problem_instance, state, action, num_action)
            
            state = next_state
            action_seq += [action]
        
        return action_seq
            
    def _generate_adaptations(self, problem_instance):
        self.epsilon = self.epsilon_start
        self.weights = np.zeros(self.num_features)
                
        self._init_wandb(problem_instance) 
        self._train(problem_instance)
        adaptation = self._get_final_adaptations(problem_instance)
        
        wandb.log({"Adaptation": adaptation})
        wandb.finish()
        
        return adaptation
    

class HallucinatedLFA(BasicLFA):
    def __init__(self, env, rng, percent_obstacles):
        super(HallucinatedLFA, self).__init__(env, rng, percent_obstacles)
        self.max_seq_len = 6
        
    def _sample_permutations(self, action_seq):
        permutations = {}
        permutations[tuple(action_seq)] = None
        
        tmp_action_seq = action_seq.copy()
        has_terminating_action = self.action_dims - 1 in tmp_action_seq
        terminating_action = tmp_action_seq.pop() if has_terminating_action else None
        
        if len(tmp_action_seq) <= self.max_seq_len:
            tmp_permutations = itertools.permutations(tmp_action_seq)
            for permutation in tmp_permutations:
                if terminating_action is not None:
                    permutations[tuple(permutation) + (terminating_action,)] = None
                else:
                    permutations[tuple(permutation)] = None
        else:
            i = 1
            max_samples = math.factorial(self.max_seq_len)
            while i < max_samples:
                random.shuffle(tmp_action_seq)
                if terminating_action is not None:
                    permutations[tuple(tmp_action_seq) + (terminating_action,)] = None
                else:
                    permutations[tuple(tmp_action_seq)] = None
                i += 1   
        
        return list(permutations.keys()) 
    
    # Hallucinate episodes by permuting the action sequence to simulate commutativity
    def _hallucinate(self, problem_instance, start_state, action_seq):        
        permutations = self._sample_permutations(action_seq)
        for permutation in permutations:
            num_action = 0
            episode_reward = 0
            state = start_state
            for action in permutation:
                num_action += 1
                reward, next_state, done = self._step(problem_instance, state, action, num_action)
                    
                self._update_weights(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
    
    def _train(self, problem_instance):
        rewards = []
        
        for _ in range(self.num_episodes):
            done = False
            num_action = 0
            action_seq = []
            episode_reward = 0
            start_state = self._generate_state(problem_instance)
            state = start_state
            while not done:
                num_action += 1
                action = self._select_action(state)
                reward, next_state, done = self._step(problem_instance, state, action, num_action)   
                             
                state = next_state
                action_seq += [action]
                episode_reward += reward
                
            self._hallucinate(problem_instance, start_state, action_seq)
            self.epsilon *= self.epsilon_decay
            
            rewards.append(episode_reward)
            avg_rewards = np.mean(rewards[-self.sma_window:])
            wandb.log({"Average Reward": avg_rewards}) 
            
    
class CommutativeLFA(BasicLFA):
    def __init__(self, env, rng, percent_obstacles):
        super(CommutativeLFA, self).__init__(env, rng, percent_obstacles) 
        
    def _get_state_idx(self, state):
        tmp_state = state.reshape(-1)
        binary_str = "".join(str(cell) for cell in reversed(tmp_state))
        state_idx = int(binary_str, 2)
        
        self._get_state(state_idx)
        return state_idx   
    
    def _get_state(self, state_idx):
        binary_str = format(state_idx, f'0{self.grid_dims[0] * self.grid_dims[1]}b')
        state = np.zeros(self.grid_dims, dtype=int)
        for i, bit in enumerate(reversed(binary_str)):
            row = i // self.num_cols
            col = i % self.num_cols
            state[row, col] = int(bit)  
        return state
    
    def _update_weights(self, state, action, reward, next_state, done):
        super()._update_weights(state, action, reward, next_state, done)
        
        if self.previous_sample is None:
            s = self._get_state_idx(state)
            a = action
            r_0 = reward
            s_1 = self._get_state_idx(next_state)
            
            self.previous_sample = (s, a, r_0)
            self.ptr_lst[(s, a)] = (s_1, r_0)
        else:
            s, a, r_0 = self.previous_sample
            s_1 = self._get_state_idx(state)
            b = action
            r_1 = reward
            s_prime = next_state
                        
            if (s, b) in self.ptr_lst:
                s_2, r_2 = self.ptr_lst[(s, b)]
                s_2 = self._get_state(s_2)
                
                _, max_next_q_value = self._get_max_q_value(s_prime)
                
                current_features = self._extract_features(s_2, a)
                current_q_value = np.dot(self.weights, current_features)
                
                td_target = r_0 - r_2 + r_1 + (1 - done) * max_next_q_value
                td_error = td_target - current_q_value
                
                self.weights = self.weights + self.alpha * (td_error * current_features)
            
            s_prime = self._get_state_idx(s_prime)
            self.previous_sample = (s_1, b, r_1)
            self.ptr_lst[(s_1, b)] = (s_prime, r_1)
    
        if done:
            self.previous_sample = None
            
    def _generate_adaptations(self, problem_instance):
        # (s, a) -> (s', r)
        self.ptr_lst = {}
        self.previous_sample = None
        
        return super()._generate_adaptations(problem_instance)