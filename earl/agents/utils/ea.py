import os
import copy
import pickle
import problems
import numpy as np
import networkx as nx

from itertools import chain
from sklearn.preprocessing import OneHotEncoder

class EA:
    def __init__(self, env, num_obstacles):
        self.max_actions = 6
        self.action_set = set()
        self.mapping = {'F': 0, 'H': 1, 'S': 2, 'G': 3, 'T': 4}
        
        self.env = env
        self.configs_to_consider = 500
        self.num_cols = env.unwrapped.ncol
        self.num_obstacles = num_obstacles
        self.num_actions = env.observation_space.n
        self.state_dims = env.observation_space.n * len(self.mapping)
        
        self.rng = np.random.default_rng(seed=42)
        self.encoder = OneHotEncoder(categories=[range(len(self.mapping))])
        
    def _save(self, approach, problem_instance, adaptation):
        directory = f'earl/agents/history/{approach.lower()}'
        filename = f'{problem_instance}.pkl'
        file_path = os.path.join(directory, filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(file_path, 'wb') as file:
            pickle.dump(adaptation, file)
            
    def _load(self, approach, problem_instance):
        directory = f'earl/agents/history/{approach}'
        filename = f'{problem_instance}.pkl'
        file_path = os.path.join(directory, filename)
        with open(file_path, 'rb') as f:
            adaptation = pickle.load(f)
        return adaptation
        
    # Convert map description to encoded array
    def _convert_state(self, state):
        flattened_state = np.array(list(chain.from_iterable(state)))
        numerical_state = np.vectorize(self.mapping.get)(flattened_state)
        return numerical_state
    
    # Calculate rewards for a given configuration by averaging A* path lengths over multiple trials
    def _get_reward(self, problem_instance, configuration):
        def manhattan_dist(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        rewards = []
        desc = copy.deepcopy(configuration).reshape((8, 8))
        row_idx, col_idx = np.where(desc == self.mapping['T'])
        transporters = set(zip(row_idx, col_idx))
        
        for _ in range(self.configs_to_consider):
            graph = nx.grid_graph(dim=[self.num_cols, self.num_cols])
            start, goal, obstacles = problems.get_entity_positions(problem_instance, self.num_obstacles)
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
    def _step(self, problem_instance, state, action, num_actions):
        reward = 0
        done = False   
        state = copy.deepcopy(state)
        prev_num_actions = len(self.action_set)
        
        next_state = state
        next_state[action] = self.mapping['T']
        self.action_set.add(action)
        
        if len(self.action_set) == prev_num_actions or num_actions == self.max_actions:
            done = True
            reward = self._get_reward(problem_instance, next_state)  
            self.action_set.clear()  
        
        return reward, next_state, done
    
    # Get adaptations for a given problem instance
    def get_adaptations(self, problem_instance):
        losses = None
        rewards = None
        approach = self.__class__.__name__
        try:
            adaptations = self._load(approach, problem_instance)
        except FileNotFoundError:
            print(f'No stored {approach} adaptation for {problem_instance.capitalize()} problem instance.')
            print('Generating new adaptation...\n')
            adaptations, losses, rewards = self._generate_adaptations(problem_instance)
            self._save(approach, problem_instance, adaptations)
        
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