import math
import wandb
import random
import itertools
import numpy as np

from agents.utils.ea import EA


class BasicQTable(EA):
    def __init__(self, env, rng, random_state):
        super(BasicQTable, self).__init__(env, rng, random_state)
                
        self.q_table = None
        # Add a dummy action (+1) to terminate the episode
        self.nS = 2 ** 16
        
        self.alpha = 0.001
        self.max_seq_len = 4
        self.epsilon_start = 1
        self.epsilon_decay = 0.99
        self.sma_window = 1000 if random_state else 500
        self.num_episodes = 25000 if random_state else 5000
        
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
        
    def _select_action(self, state):
        if self.rng.random() < self.epsilon:
            original_action = self.rng.integers(self.action_dims)
        else:
            state_idx = self._get_state_idx(state)
            original_action = self.q_table[state_idx].argmax()
            
        transformed_action = self._transform_action(original_action)
        return transformed_action, original_action
    
    def _update_q_table(self, state, action, reward, next_state, done):
        a = action
        s = self._get_state_idx(state)
        s_prime = self._get_state_idx(next_state)
        
        td_target = reward + (1 - done) * self.q_table[s_prime].max() 
        td_error = td_target - self.q_table[s, a]
        
        self.q_table[s, a] += self.alpha * td_error
    
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
                
                self._update_q_table(state, original_action, reward, next_state, done)
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
            
            return best_action_seq, best_reward
        
    def _generate_adaptations(self, problem_instance):
        self.epsilon = self.epsilon_start
        self.q_table = np.zeros((self.nS, self.action_dims))
        
        self._set_max_action(problem_instance)
        self._init_wandb(problem_instance)
        best_adaptation, best_reward = self._train(problem_instance)
        
        if self.random_state:
            best_adaptation, best_reward = self._get_final_adaptation(problem_instance)
        
        wandb.log({'Adaptation': best_adaptation})
        wandb.log({'Final Reward': best_reward})
        wandb.finish()
        
        return best_adaptation
    
    
class CommutativeQTable(BasicQTable):
    def __init__(self, env, rng, random_state):
        super(CommutativeQTable, self).__init__(env, rng, random_state)
    
    """
    Update Rule 0: Traditional Q-Update
    Q(s_1, b) = Q(s_1, b) + alpha * (r_1 + * max_a Q(s', a) - Q(s_1, b))
    Update Rule 1: Commutative Q-Update
    Q(s_2, a) = Q(s_2, a) + alpha * (r_0 - r_2 + r_1 + max_a Q(s', a) - Q(s_2, a))
    """
    def _update_q_table(self, state, action, reward, next_state, done):
        super()._update_q_table(state, action, reward, next_state, done)
        
        state_idx = self._get_state_idx(state)
        self.ptr_lst[(state_idx, action)] = (reward, next_state)
        
        if self.previous_sample is None:
            self.previous_sample = (state_idx, action, reward)
        else:
            prev_state_idx, prev_action, prev_reward = self.previous_sample
            
            if (prev_state_idx, action) in self.ptr_lst:
                r_2, s_2 = self.ptr_lst[(prev_state_idx, action)]
                r_3 = prev_reward + reward - r_2
                
                super()._update_q_table(s_2, prev_action, r_3, next_state, done)
                
            self.previous_sample = (state_idx, action, reward)
            self.ptr_lst[(state_idx, action)] = (reward, next_state)
            
        if done:
            self.previous_sample = None
            
    def _generate_adaptations(self, problem_instance):
        self.ptr_lst = {}
        self.previous_sample = None
        
        return super()._generate_adaptations(problem_instance)


class HallucinatedQTable(BasicQTable):
    def __init__(self, env, rng, random_state):
        super(HallucinatedQTable, self).__init__(env, rng, random_state)
                
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
            state = start_state
            for original_action in permutation:
                num_action += 1
                transformed_action = self._transform_action(original_action)
                reward, next_state, done = self._step(problem_instance, state, transformed_action, num_action)
                    
                self._update_q_table(state, original_action, reward, next_state, done)
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
                
            self._hallucinate(problem_instance, start_state, action_seq)
            self.epsilon *= self.epsilon_decay
            
            rewards.append(episode_reward)
            avg_rewards = np.mean(rewards[-self.sma_window:])
            wandb.log({"Average Reward": avg_rewards})
            
            if episode_reward > best_reward:
                best_action_seq = action_seq
                best_reward = episode_reward
            
        return best_action_seq, best_reward