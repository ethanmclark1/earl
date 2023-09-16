import os
import copy
import torch
import wandb
import pickle
import numpy as np
import networkx as nx

from agents.utils.network import DuelingDQN

class EARL:
    def __init__(self, env, num_obstacles):
        self._init_hyperparams()
                
        self.env = env
        self.buffer = None
        self.num_actions = 64
        self.max_adaptive_actions = 8
        self.configs_to_consider = 250
        self.num_obstacles = num_obstacles
        self.rng = np.random.default_rng(seed=42)
        
        state_dims = env.observation_space.n
        
    def _save(problem_instance, reconfiguration):
        directory = 'earl/agents/history'
        filename = f'{problem_instance}'
        file_path = os.path.join(directory, filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(file_path, 'wb') as file:
            pickle.dump(reconfiguration, file)
            
    def _load(self, problem_instance):
        directory = 'earl/agents/history'
        filename = f'{problem_instance}.pkl'
        file_path = os.path.join(directory, filename)
        with open(file_path, 'rb') as f:
            reconfiguration = pickle.load(f)
        return reconfiguration
    
    def _init_hyperparams(self):
        num_records = 10
        
        self.tau = 5e-3
        self.alpha = 1e-4
        self.gamma = 0.99
        self.batch_size = 256
        self.granularity = 0.20
        self.memory_size = 30000
        self.epsilon_start = 1.0
        self.dummy_episodes = 200
        self.num_episodes = 15000
        self.epsilon_decay = 0.9997
        self.record_freq = self.num_episodes // num_records
        
    def _init_wandb(self, problem_instance):
        wandb.init(project='earl', entity='ethanmclark1', name=problem_instance)
        config = wandb.config
        config.tau = self.tau
        config.alpha = self.alpha
        config.gamma = self.gamma 
        config.epsilon = self.epsilon_start
        config.batch_size = self.batch_size
        config.granularity = self.granularity
        config.memory_size = self.memory_size
        config.num_episodes = self.num_episodes
        config.epsilon_decay = self.epsilon_decay
        config.dummy_episodes = self.dummy_episodes
        
    def _decrement_exploration(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(0.01, self.epsilon)
        
    def _select_action(self, state):
        with torch.no_grad():
            if np.random.random() < self.epsilon:
                action = np.random.randint(len(self.num_actions))
            else:
                q_vals = self.dqn(torch.tensor(state))
                action = torch.argmax(q_vals).item()
        return action
    
    def _step(self, problem_instance, action, num_adaptive_actions):
        reward = 0
        done = False   
        
        reconfiguration = copy.deepcopy(self.env.unwrapped.desc)
        reconfiguration[action] = b'F'
        
        if num_adaptive_actions == self.max_adaptive_actions:
            done = True
            reward = self._get_reward(reconfiguration, problem_instance)    
        
        return reward, next_state, done, reconfiguration
    
    def _populate_buffer(self, problem_instance):
        for _ in range(self.dummy_episodes):
            done = False
            num_adaptive_actions = 0
            while not done:
                num_adaptive_actions += 1
                action = self._select_action()
                reward, next_state, done, _ = self._step(problem_instance, action, num_adaptive_actions)
                self.buffer.add((state, action, reward, next_state, done))
                num_adaptive_actions += 1
                
    def _get_reward(self, reconfiguration, problem_instance):
        a=3
        
    def _learn(self):
        a=3
                
    def _train(self, problem_instance):
        losses = []
        returns = []
        best_adaptive_actions = None
        best_reward = -np.inf
        
        for episode in range(self.num_episodes):
            done = False
            num_adaptive_actions = 0
            action_lst = []
            while not done:
                num_adaptive_actions += 1
                action = self._select_action(state)
                reward, next_state, done, _ = self._step(problem_instance, action, num_adaptive_actions)
                self.buffer.add((state, action, reward, next_state, done))
                loss, td_error, tree_idxs = self._learn()
                state = next_state
                action_lst.append(action)
                
            self.decrement_epsilon()
            
            losses.append(loss)
            returns.append(reward)
            avg_loss = np.mean(losses[-100:])
            avg_returns = np.mean(returns[-100:])
            
            if reward > best_reward:
                best_reward = reward
                best_adaptive_actions = action_lst
                
        return best_adaptive_actions, best_reward
    
    def _generate_reconfiguration(self, problem_instance):
        self._init_wandb(problem_instance)
        
        self.epsilon = self.epsilon_start
        self.buffer = None
        self.dqn = DuelingDQN(self.state_dims, self.num_actions, self.alpha)
        self.target_dqn = copy.deepcopy(self.dqn)
        
        self._populate_buffer(problem_instance)
        best_adaptive_actions, best_reward = self._train(problem_instance)
        
        wandb.log({'Final Reward': best_reward})
        wandb.log({'Final Adaptive Actions': best_adaptive_actions})
        wandb.finish()
        
        return best_adaptive_actions
            
    def get_reconfigured_env(self, problem_instance):
        try:
            reconfiguration = self._load(problem_instance)
        except FileNotFoundError:
            print(f'No stored reconfiguration for {problem_instance} problem instance.')
            print('Generating new reconfiguration...\n')
            reconfiguration = self._generate_reconfiguration(problem_instance)
            self._save(problem_instance, reconfiguration)
        
        return reconfiguration