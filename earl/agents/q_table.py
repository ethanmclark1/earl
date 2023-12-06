import math
import copy
import wandb
import random
import itertools
import collections
import numpy as np

from agents.utils.ea import EA


class BasicQTable(EA):
    def __init__(self, env, problem_space, num_obstacles):
        super(BasicQTable, self).__init__(env, problem_space, num_obstacles)
        
        self.q_table = None
        self.nS = 2 ** self.state_dims
        
        self.alpha = 0.0005
        self.epsilon_start = 1
        self.num_episodes = 10000
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

    def _get_state_idx(self, state):
        binary_str = "".join(str(cell) for cell in reversed(state))
        state_idx = int(binary_str, 2)
        return state_idx   
        
    def _select_action(self, state):
        if self.rng.random() < self.epsilon:
            action = self.rng.integers(self.action_dims)
        else:
            state_idx = self._get_state_idx(state)
            action = self.q_table[state_idx].argmax()
        return action
    
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
            action_seq = []
            state = np.zeros(self.state_dims, dtype=int)
            while not done:
                num_action += 1
                action = self._select_action(state)
                reward, next_state, done = self._step(problem_instance, state, action, num_action)
                
                self._update_q_table(state, action, reward, next_state, done)
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
        self.q_table = np.zeros((self.nS, self.action_dims))
        
        self._train(problem_instance)
        adaptation, reward = self._get_adaptation(problem_instance)
        
        wandb.log({'Adaptation': adaptation})
        wandb.log({'Reward': reward})
        wandb.finish()
        
        return adaptation


class HallucinatedQTable(BasicQTable):
    def __init__(self, env, problem_space, num_obstacles):
        super(HallucinatedQTable, self).__init__(env, problem_space, num_obstacles)
        
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
                    self._update_q_table(state, action, reward, next_state, True)
                else:
                    # Transform action to fit it into the designated problem space
                    transformed_action = self._transform_action(action)
                    next_state[transformed_action] = 1
                    self._update_q_table(state, action, 0, next_state, False)
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
    
    
class CommutativeQTable(BasicQTable):
    def __init__(self, env, problem_space, num_obstacles):
        super(CommutativeQTable, self).__init__(env, problem_space, num_obstacles)
        
        self.previous_sample = None
        self.ptr_lst = {}
        
    def _update_q_table(self, state, action, reward, next_state, done):
        state_idx = self._get_state_idx(state)
        action_idx = action
        next_state_idx = self._get_state_idx(next_state)
        
        # Update Rule 0: Q-Update
        td_target = reward + (1 - done) * self.q_table[next_state_idx].max()
        td_error = td_target - self.q_table[state_idx, action_idx]
        
        self.q_table[state_idx, action_idx] = self.q_table[state_idx, action_idx] + self.alpha * td_error
        
        # TODO: Need to get r_0, r_1, r_2
        # Update Rule 1: Commutative Update
        if self.previous_sample is not None:
            prev_state_idx, prev_action, prev_reward = self.previous_sample
            if (prev_state_idx, prev_action) in self.ptr_lst:
                next_state_idx, reward = self.ptr_lst[(prev_state_idx, prev_action)]
                
                td_target = reward + (1 - done) * self.q_table[next_state_idx].max()
                td_error = td_target - self.q_table[prev_state_idx, prev_action]
                
                self.q_table[prev_state_idx, prev_action] = self.q_table[prev_state_idx, prev_action] + self.alpha * td_error
        
        if not done:
            self.previous_sample = (state_idx, action, reward)
            self.ptr_lst[(state_idx, action)] = (next_state_idx, reward)
        else:
            self.previous_sample = None
    