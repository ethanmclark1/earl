import math
import wandb
import random
import itertools
import numpy as np

from agents.utils.ea import EA

# Linear Function Approximation
class BasicLFA(EA):
    def __init__(self, env, rng, random_state, reward_prediction_type=None):
        super(BasicLFA, self).__init__(env, rng, random_state)
        self.random_state = random_state
        
        self.weights = None
        # Add a dummy action (+1) to terminate the episode
        self.num_features = (2*self.state_dims) + self.action_dims
        
        self.alpha = 0.001
        self.max_seq_len = 4
        self.epsilon_start = 1
        self.epsilon_decay = 0.99
        self.sma_window = 1000 if random_state else 250
        self.num_episodes = 10000 if random_state else 2500
        self.reward_prediction_type = reward_prediction_type

    def _init_wandb(self, problem_instance):
        config = super()._init_wandb(problem_instance)
        config.alpha = self.alpha
        config.sma_window = self.sma_window
        config.max_action = self.max_action
        config.action_cost = self.action_cost
        config.max_seq_len = self.max_seq_len
        config.random_state = self.random_state
        config.num_episodes = self.num_episodes
        config.epsilon_decay = self.epsilon_decay
        config.percent_holes = self.percent_holes
        config.action_success_rate = self.action_success_rate
        config.configs_to_consider = self.configs_to_consider
        config.reward_prediction_type = self.reward_prediction_type
        
    def _select_action(self, state):
        if self.rng.random() < self.epsilon:
            original_action = self.rng.integers(self.action_dims)
        else:
            original_action, _ = self._get_max_q_value(state)
            
        transformed_action = self._transform_action(original_action)
        return transformed_action, original_action
        
    # Feature vector is a one hot encoding of the state and the action
    def _get_features(self, state, action):
        state_features = np.empty(2*self.state_dims)
        
        mutable_state = state[2:6, 2:6]
        for i, row in enumerate(mutable_state):
            for j, cell in enumerate(row):
                index = (i * 4 + j) * 2
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
            features = self._get_features(state, action)
            q_value = np.dot(self.weights, features)
            
            if q_value > max_q_value:
                best_action = action
                max_q_value = q_value
        
        return best_action, max_q_value
    
    def _update_weights(self, state, action, reward, next_state, done):
        features = self._get_features(state, action)
        current_q_value = np.dot(self.weights, features)

        _, max_next_q_value = self._get_max_q_value(next_state)
        td_target = reward + (1 - done) * max_next_q_value
        td_error = td_target - current_q_value

        self.weights += self.alpha * td_error * features

    def _train(self, problem_instance):
        rewards = []
        
        best_reward = -np.inf
        best_action_seq = None
        
        for _ in range(self.num_episodes):
            done = False
            num_action = 0
            action_seq = []
            episode_reward = 0
            state = self._get_state(problem_instance)
            while not done:
                num_action += 1
                transformed_action, original_action = self._select_action(state)
                reward, next_state, done = self._step(problem_instance, state, transformed_action, num_action)
                
                self._update_weights(state, original_action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                action_seq += [original_action]
                
            self.epsilon *= self.epsilon_decay
            
            rewards.append(episode_reward)
            avg_rewards = np.mean(rewards[-self.sma_window:])
            wandb.log({"Average Reward": avg_rewards})
            
            if episode_reward > best_reward:
                best_action_seq = action_seq
                best_reward = episode_reward
            
        return best_action_seq, best_reward
    
    def _get_final_adaptation(self, problem_instance):
        best_reward = -np.inf
        best_action_seq = None
        
        for _ in range(25):
            done = False
            num_action = 0
            action_seq = []
            self.epsilon = 0
            episode_reward = 0
            state = np.zeros(self.grid_dims, dtype=int)
            while not done:
                num_action += 1
                transformed_action, original_action = self._select_action(state)
                reward, next_state, done = self._step(problem_instance, state, transformed_action, num_action)
                
                state = next_state
                action_seq += [original_action]
                episode_reward += reward
                
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_action_seq = action_seq
            
            return best_action_seq
            
    def _generate_adaptations(self, problem_instance):
        self.epsilon = self.epsilon_start
        self.weights = np.zeros(self.num_features)
        
        self._set_max_action(problem_instance)    
        self._init_wandb(problem_instance) 
        best_adaptation, best_reward = self._train(problem_instance)
        
        if self.random_state:
            best_adaptation, best_reward = self._get_final_adaptation(problem_instance)
        
        wandb.log({'Adaptation': best_adaptation})
        wandb.log({'Final Reward': best_reward})
        wandb.finish()
        
        return best_adaptation
    
    
class CommutativeLFA(BasicLFA):
    def __init__(self, env, rng, random_state, reward_prediction_type):
        super(CommutativeLFA, self).__init__(env, rng, random_state, reward_prediction_type) 
        
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
    
    def _update_weights(self, state, action, reward, next_state, done, num_action):
        super()._update_weights(state, action, reward, next_state, done, num_action)
        
        state_idx = self._get_state_idx(state)
        self.ptr_lst[(state_idx, action)] = (reward, next_state)
        
        if self.previous_sample is None:
            self.previous_sample = (state_idx, action, reward)
        else:
            prev_state_idx, prev_action, prev_reward = self.previous_sample
       
            if (prev_state_idx, action) in self.ptr_lst:
                r_2, s_2 = self.ptr_lst[(prev_state_idx, action)]
                r_3 = prev_reward + reward - r_2
                
                super()._update_weights(s_2, prev_action, r_3, next_state, done)
            
            self.previous_sample = (state_idx, action, reward)
            self.ptr_lst[(state_idx, action)] = (reward, next_state)
    
        if done:
            self.previous_sample = None
            
    def _generate_adaptations(self, problem_instance):
        self.ptr_lst = {}
        self.previous_sample = None
        
        return super()._generate_adaptations(problem_instance)
    

class HallucinatedLFA(BasicLFA):
    def __init__(self, env, rng, random_state):
        super(HallucinatedLFA, self).__init__(env, rng, random_state)
        
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
    def _hallucinate(self, start_state, action_seq, episode_reward):
        permutations = self._sample_permutations(action_seq)
        for permutation in permutations:
            num_action = 0
            state = start_state
            terminating_action = permutation[-1]
            for original_action in permutation:
                num_action += 1
                transformed_action = self._transform_action(original_action)
                next_state = self._get_next_state(state, transformed_action)
                
                if original_action == terminating_action:
                    reward = episode_reward
                    done = True
                else:
                    reward = 0
                    done = False
                    
                self._update_weights(state, original_action, reward, next_state, done)
                state = next_state
    
    def _train(self, problem_instance):
        rewards = []
        
        best_reward = -np.inf
        best_action_seq = None
        
        for _ in range(self.num_episodes):
            done = False
            num_action = 0
            action_seq = []
            episode_reward = 0
            start_state = self._get_state(problem_instance)
            state = start_state
            while not done:
                num_action += 1
                transformed_action, original_action = self._select_action(state)
                reward, next_state, done = self._step(problem_instance, state, transformed_action, num_action)   
                             
                state = next_state
                action_seq += [original_action]
                episode_reward += reward
                
            self._hallucinate(start_state, action_seq, episode_reward)
            self.epsilon *= self.epsilon_decay
            
            rewards.append(episode_reward)
            avg_rewards = np.mean(rewards[-self.sma_window:])
            wandb.log({"Average Reward": avg_rewards}) 
            
            if episode_reward > best_reward:
                best_action_seq = action_seq
                best_reward = episode_reward
            
        return best_action_seq, best_reward