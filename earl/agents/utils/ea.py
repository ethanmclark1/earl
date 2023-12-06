import os
import copy
import wandb
import pickle
import problems
import numpy as np
import networkx as nx


class EA:
    def __init__(self, env, problem_space, num_obstacles):
        self._init_hyperparams()
        
        self.env = env
        self.problem_space = problem_space
        self.num_cols = env.unwrapped.ncol
        self.num_obstacles = num_obstacles
        
        self.grid_dims = env.unwrapped.desc.shape
        self.grid_size = '4x4' if self.grid_dims[0] == 4 else '8x8'
        self.state_dims = env.observation_space.n
        # Add a dummy action (+1) to terminate the episode
        self.action_dims = 4 + 1 if problem_space == 'small' else env.observation_space.n + 1
        
        self.random_seed = 42
        self.rng = np.random.default_rng(seed=self.random_seed)
        
    def _init_hyperparams(self):
        self.action_cost = 0.50
        self.sma_percentage = 0.05
        self.configs_to_consider = 100
        self.action_success_rate = 0.25

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
    
    def _transform_action(self, action):
        if self.problem_space == 'small':
            # Shift action indices to the middle 4 cells of the grid
            # 0 -> 5; 1 -> 6; 2 -> 9; 3 -> 10
            shift = 5
            if action == 2 or action == 3:
                shift += 2
            action += shift
        return action     
    
    def _create_graph(self):
        graph = nx.grid_graph(dim=[self.num_cols, self.num_cols])
        nx.set_edge_attributes(graph, 1, 'weight')
        return graph
    
    # Calculate rewards for a given configuration by averaging A* path lengths over multiple trials
    """
    Cell Values:
        0: Safe
        1: Bridge
        2: Start
        3: Goal
        4: Hole
    """
    def _get_reward(self, problem_instance, configuration, num_action):
        def manhattan_dist(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        bridge_bonus = 2
        valid_path_reward = 20
        invalid_path_penalty = -20
        path_length_penalty = 1

        rewards = []
        desc = copy.deepcopy(configuration).reshape(self.grid_dims)
        row_idx, col_idx = np.where(desc == 1)
        bridges = set(zip(row_idx, col_idx))
        
        paths = []
        for _ in range(self.configs_to_consider):
            path = None
            graph = self._create_graph()
            start, goal, obstacles = problems.get_entity_positions(problem_instance, self.grid_size, self.num_obstacles)
            
            # Bridges cannot cover start or goal cells 
            if desc[start] == 1 or desc[goal] == 1:
                rewards += [invalid_path_penalty]
                continue
            
            desc[start], desc[goal] = 2, 3
            # Populate obstacle in cell only if bridge is not already there
            for obstacle in obstacles:
                obstacle = tuple(obstacle)
                if desc[obstacle] != 1:
                    desc[obstacle] = 4
                    graph.remove_node(obstacle)
            
            for bridge in bridges:
                bridge = tuple(bridge)
                for neighbor in graph.neighbors(bridge):
                    graph[bridge][neighbor]['weight'] = 0
            
            try:          
                path = nx.astar_path(graph, start, goal, manhattan_dist, 'weight')
                reward = valid_path_reward - (path_length_penalty * len(path))
                bridges_used = len(set(path) & bridges)
                reward += (bridge_bonus * bridges_used)
                action_penalty = self.action_cost * num_action
                reward -= action_penalty
                rewards += [reward]
            except nx.NetworkXNoPath:
                rewards += [invalid_path_penalty]
                
            desc = copy.deepcopy(configuration).reshape(self.grid_dims)

        avg_reward = np.mean(rewards)
        return avg_reward
    
    # Apply adaptation to task environment
    def _step(self, problem_instance, state, action, num_action):
        reward = 0
        state = copy.deepcopy(state)
        terminating_action = self.action_dims - 1
        
        next_state = state
        # Add stochasticity to instrumental action execution
        if action != terminating_action and self.action_success_rate > self.rng.random():
            action = self._transform_action(action)
            next_state[action] = 1
        
        done = action == terminating_action
        if done:
            reward = self._get_reward(problem_instance, next_state, num_action)  
        
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