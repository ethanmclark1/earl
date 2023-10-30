import os
import copy
import wandb
import pickle
import problems
import numpy as np
import networkx as nx

from itertools import chain
from sklearn.preprocessing import OneHotEncoder


class EA:
    def __init__(self, env, num_obstacles):
        self._init_hyperparams()
        
        self.action_set = set()
        self.mapping = {'F': 0, 'H': 1, 'S': 2, 'G': 3, 'T': 4}
        
        self.env = env
        self.num_cols = env.unwrapped.ncol
        self.num_obstacles = num_obstacles
        self.state_dims = env.observation_space.n * len(self.mapping)
        self.num_actions = self.max_actions = env.observation_space.n
        
        self.rng = np.random.default_rng(seed=42)
        self.encoder = OneHotEncoder(categories=[range(len(self.mapping))])
        
    def _init_hyperparams(self):
        num_records = 10

        self.tau = 5e-3
        self.alpha = 1e-4
        self.batch_size = 256
        self.memory_size = 10000
        self.action_cost = -0.10
        self.num_episodes = 10000
        self.kl_coefficient = 1e-4
        self.configs_to_consider = 200
        self.record_freq = self.num_episodes // num_records

    def _save(self, approach, problem_instance, affinity_instance, adaptation):
        directory = f'earl/agents/history/{approach.lower()}/{problem_instance}'
        filename = f'{affinity_instance}.pkl'
        file_path = os.path.join(directory, filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(file_path, 'wb') as file:
            pickle.dump(adaptation, file)
            
    def _load(self, approach, problem_instance, affinity_instance):
        directory = f'earl/agents/history/{approach.lower()}/{problem_instance}'
        filename = f'{affinity_instance}.pkl'
        file_path = os.path.join(directory, filename)
        with open(file_path, 'rb') as f:
            adaptation = pickle.load(f)
        return adaptation
    
    def _init_wandb(self, problem_instance, affinity_instance):
        wandb.init(
            project='earl', 
            entity='ethanmclark1', 
            name=f'{self.__class__.__name__}/{problem_instance.capitalize()}/{affinity_instance.capitalize()}'
            )
        
        config = wandb.config
        return config
        
    # Convert map description to encoded array
    def _convert_state(self, state):
        flattened_state = np.array(list(chain.from_iterable(state)))
        numerical_state = np.vectorize(self.mapping.get)(flattened_state)
        return numerical_state
    
    # Calculate rewards for a given configuration by averaging A* path lengths over multiple trials
    def _get_reward(self, problem_instance, affinity_instance, configuration):
        def manhattan_dist(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        rewards = []
        desc = copy.deepcopy(configuration).reshape((8, 8))
        row_idx, col_idx = np.where(desc == self.mapping['T'])
        transporters = set(zip(row_idx, col_idx))
        
        for _ in range(self.configs_to_consider):
            graph = nx.grid_graph(dim=[self.num_cols, self.num_cols])
            start, goal, obstacles = problems.get_entity_positions(problem_instance, affinity_instance, self.num_obstacles)
            desc[start] = self.mapping['S']
            desc[goal] = self.mapping['G']
            for obstacle in obstacles:
                if desc[obstacle] != self.mapping['T']:
                    desc[obstacle] = self.mapping['H']
            
            for i in range(self.num_cols):
                for j in range(self.num_cols):
                    cell_value = desc[i, j]
                    for neighbor in graph.neighbors((i, j)):
                        weight = 6
                        if cell_value == self.mapping['H']:
                            weight = 100
                        elif cell_value == self.mapping['T']:
                            weight = 0
                        graph[(i, j)][neighbor]['weight'] = weight
                        
            path = set(nx.astar_path(graph, start, goal, manhattan_dist, 'weight'))
            rewards += [-len(path - transporters)]

        avg_reward = np.mean(rewards)
        return avg_reward
    
    # Apply adaptation to task environment
    def _step(self, problem_instance, affinity_instance, state, action, num_actions):
        reward = 0
        done = False   
        action_success_rate = 0.8
        state = copy.deepcopy(state)
        prev_num_actions = len(self.action_set)
        
        next_state = state
        # Add stochasticity to action execution
        if action_success_rate > self.rng.random():
            next_state[action] = self.mapping['T']
        self.action_set.add(action)
        
        if len(self.action_set) == prev_num_actions or num_actions == self.max_actions:
            done = True
            reward = self.action_cost * num_actions
            reward += self._get_reward(problem_instance, affinity_instance, next_state)  
            self.action_set.clear()  
        
        return reward, next_state, done
    
    # Get adaptations for a given problem instance
    def get_adaptations(self, problem_instance, affinity_instance):
        losses = None
        rewards = None
        approach = self.__class__.__name__
        try:
            adaptations = self._load(approach, problem_instance, affinity_instance)
        except FileNotFoundError:
            print(f'No stored {approach} adaptation for {problem_instance.capitalize()} problem instance combined with {affinity_instance.capitalize()} affinity instance.')
            print('Generating new adaptation...')
            adaptations, losses, rewards = self._generate_adaptations(problem_instance, affinity_instance)
            self._save(approach, problem_instance, affinity_instance, adaptations)
        
        print(f'{approach} adaptations for {problem_instance.capitalize()} problem instance combined with {affinity_instance.capitalize()} affinity instance:\n{adaptations}\n')
        return adaptations, losses, rewards
    
    # Apply adaptations to task environment
    def get_adapted_env(self, desc, adaptations):
        for reconfigruation in adaptations:
            row = reconfigruation // self.num_cols
            col = reconfigruation % self.num_cols
            if desc[row, col] == b'S' or desc[row, col] == b'G':
                continue
            desc[row, col] = b'T'
        
        return desc