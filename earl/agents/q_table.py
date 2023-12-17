import pdb
import math
import wandb
import random
import itertools
import numpy as np

from agents.utils.ea import EA


class BasicQTable(EA):
    def __init__(self, env, rng, percent_obstacles):
        super(BasicQTable, self).__init__(env, rng, percent_obstacles)
                
        self.q_table = None
        # Add a dummy action (+1) to terminate the episode
        self.action_dims = 16 + 1
        self.nS = 2 ** 16
        
        self.alpha = 0.0004
        self.epsilon_start = 1
        self.num_episodes = 50000
        self.epsilon_decay = 0.9999
        self.sma_window = int(self.num_episodes * self.sma_percentage)
        
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
    
    def _get_state_idx(self, state):
        mutable_state = state[2:6, 2:6].reshape(-1)
        binary_str = "".join(str(cell) for cell in reversed(mutable_state))
        state_idx = int(binary_str, 2)
        return state_idx   

    # Transform action so that it can be used to modify the state
    # 0 -> 18; 1 -> 19; 2 -> 20; 3 -> 21 
    # 4 -> 26; 5 -> 27; 6 -> 28; 7 -> 29
    # 8 -> 34; 9 -> 35; 10 -> 36; 11 -> 37
    # 12 -> 42; 13 -> 43; 14 -> 44; 15 -> 45
    def _transform_action(self, action):
        if action == self.action_dims - 1:
            return action

        row = action // 4
        col = action % 4

        shift = 18 + row * 8 + col
        return shift
        
    def _select_action(self, state):
        if self.rng.random() < self.epsilon:
            original_action = self.rng.integers(self.action_dims)
        else:
            state_idx = self._get_state_idx(state)
            original_action = self.q_table[state_idx].argmax()
            
        transformed_action = self._transform_action(original_action)
        return transformed_action, original_action
    
    def _update_q_table(self, state, action, reward, next_state, done):
        action_idx = action
        state_idx = self._get_state_idx(state)
        next_state_idx = self._get_state_idx(next_state)
        
        td_target = reward + (1 - done) * self.q_table[next_state_idx].max() 
        td_error = td_target - self.q_table[state_idx, action_idx]
        
        self.q_table[state_idx, action_idx] = self.q_table[state_idx, action_idx] + self.alpha * td_error
    
    def _train(self, problem_instance):
        rewards = []
        
        for _ in range(self.num_episodes):
            done = False
            num_action = 0
            episode_reward = 0
            state = self._generate_state(problem_instance)
            while not done:
                num_action += 1
                transformed_action, original_action = self._select_action(state)
                reward, next_state, done = self._step(problem_instance, state, transformed_action, num_action)
                
                self._update_q_table(state, original_action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                
            self.epsilon *= self.epsilon_decay
            
            rewards.append(episode_reward)
            avg_rewards = np.mean(rewards[-self.sma_window:])
            wandb.log({"Average Reward": avg_rewards})
                        
    def _get_final_adaptation(self, problem_instance):
        done = False
        num_action = 0
        action_seq = []
        self.epsilon = 0
        state = np.zeros(self.grid_dims, dtype=int)
        while not done:
            num_action += 1
            transformed_action, original_action = self._select_action(state)
            _, next_state, done = self._step(problem_instance, state, transformed_action, num_action)
            
            state = next_state
            action_seq += [original_action]
            
        return action_seq
        
    def _generate_adaptations(self, problem_instance):
        self.epsilon = self.epsilon_start
        self.q_table = np.zeros((self.nS, self.action_dims))
        
        self._init_wandb(problem_instance)
        self._train(problem_instance)
        adaptation = self._get_final_adaptation(problem_instance)
        
        wandb.log({'Adaptation': adaptation})
        wandb.finish()
        
        return adaptation


class HallucinatedQTable(BasicQTable):
    def __init__(self, env, rng, percent_obstacles):
        super(HallucinatedQTable, self).__init__(env, rng, percent_obstacles)
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
            for original_action in permutation:
                num_action += 1
                transformed_action = self._transform_action(original_action)
                reward, next_state, done = self._step(problem_instance, state, transformed_action, num_action)
                    
                self._update_q_table(state, original_action, reward, next_state, done)
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
    
    
class CommutativeQTable(BasicQTable):
    def __init__(self, env, rng, percent_obstacles):
        super(CommutativeQTable, self).__init__(env, rng, percent_obstacles)
        
        # (s, a) -> (s', r)
        self.ptr_lst = {}
        self.previous_sample = None
    
    """
    Update Rule 0: Traditional Q-Update
    Q(s_1, b) = Q(s_1, b) + alpha * (r_1 + * max_a Q(s', a) - Q(s_1, b))
    Update Rule 1: Commutative Q-Update
    Q(s_2, a) = Q(s_2, a) + alpha * (r_0 - r_2 + r_1 + max_a Q(s', a) - Q(s_2, a))
    """
    def _update_q_table(self, state, action, reward, next_state, done):
        super()._update_q_table(state, action, reward, next_state, done)
        
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
            s_prime = self._get_state_idx(next_state)
            
            if (s, b) in self.ptr_lst:
                s_2, r_2 = self.ptr_lst[(s, b)]
                
                td_target = r_0 - r_2 + r_1 + (1 - done) * self.q_table[s_prime].max()
                td_error = td_target - self.q_table[s_2, a]
                
                self.q_table[s_2, a] = self.q_table[s_2, a] + self.alpha * td_error
                
            self.previous_sample = (s_1, b, r_1)
            self.ptr_lst[(s_1, b)] = (s_prime, r_1)
            
        if done:
            self.previous_sample = None