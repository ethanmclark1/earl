import math
import copy
import wandb
import random
import itertools
import numpy as np

from agents.utils.ea import EA


class BasicQTable(EA):
    def __init__(self, env, grid_size, num_obstacles):
        super(BasicQTable, self).__init__(env, grid_size, num_obstacles)
        
        self.q_table = None
        self.nS = 2 ** self.state_dims
        
        self.alpha = 0.003
        self.epsilon_start = 1
        self.num_episodes = 500
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

    def _get_state_idx(self, state):
        binary_str = "".join(str(cell) for cell in state)
        state_idx = int(binary_str, 2)
        return state_idx
        
    def _select_action(self, state):
        if self.rng.random() < self.epsilon:
            action = self.rng.integers(self.action_dims)
        else:
            state_idx = self._get_state_idx(state)
            action = self.q_table(state_idx).argmax()
        return action
    
    def _update_q_table(self, state, action, reward, next_state, done):
        state_idx = self._get_state_idx(state)
        action_idx = action
        
        if done:
            td_target = reward
        else:
            td_target = reward + self.q_table[next_state].max()
            
        td_error = td_target - self.q_table[state_idx, action_idx]
        self.q_table[state_idx, action_idx] += self.alpha * td_error
    
    def _train(self, problem_instance):
        rewards = []
        best_actions = None
        best_reward = -np.inf
        
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
                
                action_seq.append(action)
                state = next_state
                
            self._decay_epsilon()
            
            rewards.append(reward)
            if reward > best_reward:
                best_reward = reward
                best_actions = action_seq
                
            wandb.log({'Episode Reward': reward})
            wandb.log({'Episode Actions': action_seq})
            
        return best_actions, best_reward
        
    def _generate_adaptations(self, problem_instance):
        # self._init_wandb(problem_instance)
        
        self.epsilon = self.epsilon_start
        self.q_table = np.zeros((self.nS, self.action_dims))
        
        best_actions, best_reward = self._train(problem_instance)
        
        wandb.log({'Final Reward': best_reward})
        wandb.log({'Final Actions': best_actions})
        wandb.finish()
        
        return best_actions


class HallucinatedQTable(BasicQTable):
    def __init__(self, env, grid_size, num_obstacles):
        super(HallucinatedQTable, self).__init__(env, grid_size, num_obstacles)
        
        self.max_seq_len = 8
        
    def _sample_permutations(self, action_seq):
        permutations = set()
        permutations.add(tuple(action_seq))
        tmp_action_seq = action_seq.copy()
        terminating_action = tmp_action_seq.pop()
        
        if len(tmp_action_seq) <= self.max_seq_len:
            permutations_no_term = itertools.permutations(tmp_action_seq)
            permutations = {permutation + (terminating_action,) for permutation in permutations_no_term}
        else:
            while len(permutations) < math.factorial(self.max_seq_len):
                random.shuffle(tmp_action_seq)
                permutations.add(tuple(tmp_action_seq + [terminating_action]))     
        
        return list(permutations)       
    
    # Hallucinate episodes by permuting the action sequence to simulate commutativity
    def _hallucinate(self, action_seq, reward):
        start_state = np.zeros(self.state_dims, dtype=int)
        permutations = self._sample_permutations(action_seq)
        
        for permutation in permutations:
            state = start_state
            for action in permutation:
                next_state = copy.deepcopy(state)
                
                done = action == len(state)
                if done:
                    self._update_q_table(state, action, reward, next_state, True)
                else:
                    next_state[action] = 1
                    self._update_q_table(state, action, 0, next_state, False)
                    state = next_state
    
    def _train(self, problem_instance):
        rewards = []
        best_actions = None
        best_reward = -np.inf
        
        for _ in range(self.num_episodes):
            done = False
            num_action = 0
            action_seq = []
            state = np.zeros(self.state_dims, dtype=int)
            while not done:
                num_action += 1
                action = self._select_action(state)
                reward, next_state, done = self._step(problem_instance, state, action, num_action)                
                action_seq.append(action)
                state = next_state
                
            self._hallucinate(action_seq, reward)
            self._decay_epsilon()
            
            rewards.append(reward)
            if reward > best_reward:
                best_reward = reward
                best_actions = action_seq
                
            wandb.log({'Episode Reward': reward})
            wandb.log({'Episode Actions': action_seq})
            
        return best_actions, best_reward
    
    
# TODO: Add commutative update rule
class CommutativeQTable(BasicQTable):
    def __init__(self, env, grid_size, num_obstacles):
        super(CommutativeQTable, self).__init__(env, grid_size, num_obstacles)
        
    def _update_q_table(self, state, action, reward, next_state, done):
        state_idx = self._get_state_idx(state)
        action_idx = action
        
        if done:
            td_target = reward
        else:
            td_target = reward + self.q_table[next_state].max()
            
        td_error = td_target - self.q_table[state_idx, action_idx]
        self.q_table[state_idx, action_idx] += self.alpha * td_error
        
    def _train(self, problem_instance):
        rewards = []
        best_actions = None
        best_reward = -np.inf
        
        for _ in range(self.num_episodes):
            done = False
            num_action = 0
            action_seq = []
            state = np.zeros(self.state_dims, dtype=int)
            while not done:
                num_action += 1
                action = self._select_action(state)
                reward, next_state, done = self._step(problem_instance, state, action, num_action)                
                action_seq.append(action)
                state = next_state
                
            self._hallucinate(action_seq, reward)
            self._decay_epsilon()
            
            rewards.append(reward)
            if reward > best_reward:
                best_reward = reward
                best_actions = action_seq
                
            wandb.log({'Episode Reward': reward})
            wandb.log({'Episode Actions': action_seq})
            
        return best_actions, best_reward
        