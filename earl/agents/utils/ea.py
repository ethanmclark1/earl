import os
import copy
import wandb
import pickle
import problems
import numpy as np
import networkx as nx


class EA:
    def __init__(self, env, rng):
        self._init_hyperparams()
        
        self.env = env
        self.rng = rng
        
        self.num_cols = env.unwrapped.ncol
        self.grid_dims = env.unwrapped.desc.shape
        self.state_dims = env.observation_space.n
        self.grid_size = '4x4' if self.grid_dims[0] == 4 else '8x8'
        
    def _init_hyperparams(self):
        self.action_cost = 0.10
        self.sma_percentage = 0.05
        self.percent_obstacles = 0.75
        self.configs_to_consider = 25
        self.action_success_rate = 0.75

    def _save(self, approach, problem_instance, adaptation):
        directory = f'earl/agents/history/{approach.lower()}'
        filename = f'{problem_instance}.pkl'
        file_path = os.path.join(directory, filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(file_path, 'wb') as file:
            pickle.dump(adaptation, file)
            
    def _load(self, approach, problem_instance):
        directory = f'earl/agents/history/{approach.lower()}'
        filename = f'{problem_instance}.pkl'
        file_path = os.path.join(directory, filename)
        with open(file_path, 'rb') as f:
            adaptation = pickle.load(f)
        return adaptation
    
    def _init_wandb(self, problem_instance):
        wandb.init(
            project='earl', 
            entity='ethanmclark1', 
            name=f'{self.__class__.__name__}/{problem_instance.capitalize()}'
            )
        
        config = wandb.config
        return config
    
    def _create_graph(self):
        graph = nx.grid_graph(dim=[self.num_cols, self.num_cols])
        nx.set_edge_attributes(graph, 1, 'weight')
        return graph
    
    # Calculate utilities for a given configuration by averaging A* path lengths over multiple trials
    """
    Cell Values:
        0: Frozen
        1: Bridge
        2: Start
        3: Goal
        4: Hole
    """
    def _calc_utility(self, problem_instance, configuration):
        def manhattan_dist(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        utilities = []
        graph = self._create_graph()
        desc = copy.deepcopy(configuration).reshape(self.grid_dims)
        row_idx, col_idx = np.where(desc == 1)
        
        bridges = set(zip(row_idx, col_idx))
        for bridge in bridges:
            bridge = tuple(bridge)
            for neighbor in graph.neighbors(bridge):
                graph[bridge][neighbor]['weight'] = 0
        
        for _ in range(self.configs_to_consider):
            tmp_desc = desc.copy()
            tmp_graph = graph.copy()
            start, goal, obstacles = problems.get_entity_positions(problem_instance, self.rng, self.grid_size, self.percent_obstacles)
            
            # Bridges cannot cover start or goal cells 
            if tmp_desc[start] == 1 or tmp_desc[goal] == 1:
                utilities += [-self.state_dims]
                continue
            
            tmp_desc[start], tmp_desc[goal] = 2, 3
            #  obstacle in cell only if bridge is not already there
            for obstacle in obstacles:
                obstacle = tuple(obstacle)
                if tmp_desc[obstacle] != 1:
                    tmp_desc[obstacle] = 4
                    tmp_graph.remove_node(obstacle)
            
            path = nx.astar_path(tmp_graph, start, goal, manhattan_dist, 'weight')
            utility = len(set(path) - bridges)
            utilities += [-utility]

        avg_utility = np.mean(utilities)
        return avg_utility
    
    # r(s,a,s') = u(s') - u(s) - c(a)
    def _get_reward(self, problem_instance, state, done, next_state, num_action):
        util_s_prime = self._calc_utility(problem_instance, next_state)
        
        # Agent selects a non-terminating action
        if not done:
            util_s = self._calc_utility(problem_instance, state)
            reward = util_s_prime - util_s - (self.action_cost * num_action)
        # Agent selects a terminating action as the only action
        elif done and num_action == 1:
            reward = util_s_prime
        # Agent selects a non-terminating action and it isn't the only action
        elif done and num_action != 1:
            reward = 0
            
        return reward
        
    # Apply adaptation to task environment
    def _step(self, problem_instance, state, action, num_action):
        state = copy.deepcopy(state)
        terminating_action = self.action_dims - 1
        
        next_state = state.copy()
        # Add stochasticity to instrumental action execution
        if action != terminating_action and self.action_success_rate > self.rng.random():
            next_state[action] = 1
        
        done = action == terminating_action
        reward = self._get_reward(problem_instance, state, done, next_state, num_action)  
        
        return reward, next_state, done
    
    # Get adaptations for a given problem instance
    def get_adaptations(self, problem_instance):
        approach = self.__class__.__name__
        try:
            adaptations = self._load(approach, problem_instance)
        except FileNotFoundError:
            print(f'No stored adaptation for {approach} on the {problem_instance.capitalize()} problem instance.')
            print('Generating new adaptation...')
            adaptations = self._generate_adaptations(problem_instance)
            self._save(approach, problem_instance, adaptations)
        
        print(f'{approach} adaptations for {problem_instance.capitalize()} problem instance:\n{adaptations}\n')
        return adaptations
    
    # Apply adaptations to task environment
    def get_adapted_env(self, desc, adaptations):
        for reconfigruation in adaptations:
            row = reconfigruation // self.num_cols
            col = reconfigruation % self.num_cols
            if desc[row, col] == b'S' or desc[row, col] == b'G':
                continue
            desc[row, col] = b'T'
        
        return desc