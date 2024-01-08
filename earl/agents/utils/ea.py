import os
import copy
import wandb
import pickle
import problems
import numpy as np
import networkx as nx


class EA:
    def __init__(self, env, rng, random_state):
        self._init_hyperparams()
        
        self.env = env
        self.rng = rng
        self.random_state = random_state

        self.state_dims = 16
        self.action_dims = self.state_dims + 1        
        self._get_state = self._generate_state if random_state else self._generate_fixed_state
        
        self.num_cols = env.unwrapped.ncol
        self.grid_dims = env.unwrapped.desc.shape
    
    # Tabular solutions have smaller solution spaces than approximate solutions
    def _init_hyperparams(self):
        self.action_cost = 0.05
        self.percent_holes = 0.75
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
        problem_instance = 'cheese'
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
    
    # Set the maximum number of actions that can be taken in a single episode
    # according to the problem instance
    def _set_max_action(self, problem_instance):
        if problem_instance == 'cross':
            self.max_action = 10
        elif problem_instance == 'twister':
            self.max_action = 12
    
    def _generate_fixed_state(self, problem_instance):
        return np.zeros(self.grid_dims, dtype=int)
    
    # Generate initial state for a given problem instance
    def _generate_state(self, problem_instance):
        houses = problems.problems[problem_instance]['houses']
        num_bridges = self.rng.choice(self.max_action)
        bridges = self.rng.choice(self.action_dims-1, size=num_bridges, replace=True)
        state = np.zeros(self.grid_dims, dtype=int)
        
        for bridge in bridges:
            if hasattr(self, '_transform_action'):
                bridge = self._transform_action(bridge)
            row = bridge // self.num_cols
            col = bridge % self.num_cols
            bridge = (row, col)
            if bridge in houses:
                continue
            state[bridge] = 1
            
        return state
    
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
    
    def _place_bridge(self, state, action):
        next_state = copy.deepcopy(state)
        row = action // self.num_cols
        col = action % self.num_cols
        bridge = (row, col)
        next_state[bridge] = 1
        return next_state
    
    def _create_graph(self):
        graph = nx.grid_graph(dim=[self.num_cols, self.num_cols])
        nx.set_edge_attributes(graph, 10, 'weight')
        return graph
    
    # Calculate utility for a given state by averaging A* path lengths over multiple trials
    """
    Cell Values:
        0: Frozen
        1: Bridge
        2: Start
        3: Goal
        4: Hole
    """
    def _calc_utility(self, problem_instance, state):
        def manhattan_dist(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        utilities = []
        graph = self._create_graph()
        desc = copy.deepcopy(state).reshape(self.grid_dims)
        row_idx, col_idx = np.where(desc == 1)
        
        bridges = set(zip(row_idx, col_idx))
        for bridge in bridges:
            bridge = tuple(bridge)
            for neighbor in graph.neighbors(bridge):
                graph[bridge][neighbor]['weight'] = 0
        
        for _ in range(self.configs_to_consider):
            tmp_desc = copy.deepcopy(desc)
            tmp_graph = copy.deepcopy(graph)
            start, goal, obstacles = problems.get_entity_positions(problem_instance, self.rng, self.percent_holes)
            
            # Bridges cannot cover start or goal cells 
            if tmp_desc[start] == 1 or tmp_desc[goal] == 1:
                utilities += [-self.num_cols*2]
                continue
            
            tmp_desc[start], tmp_desc[goal] = 2, 3
            #  Place obstacle in cell only if bridge is not already there
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
    def _get_reward(self, problem_instance, state, action, next_state, num_action):
        reward = 0
        terminating_action = self.action_dims - 1
        
        done = action == terminating_action
        timeout = num_action == self.max_action
        
        util_s = self._calc_utility(problem_instance, state)
        if not done:
            if not np.array_equal(state, next_state):
                util_s_prime = self._calc_utility(problem_instance, next_state)
                reward = util_s_prime - util_s
            # If state == next state then u(s') - u(s) = 0
            reward -= self.action_cost * num_action
        elif done and num_action == 1:
            reward = util_s
        
        return reward, (done or timeout)
        
    # Apply adaptation to task environment
    def _step(self, problem_instance, state, action, num_action):
        next_state = copy.deepcopy(state)
        terminating_action = self.action_dims - 1
        
        # Add stochasticity to actions
        if action != terminating_action and self.action_success_rate > self.rng.random():
            next_state = self._place_bridge(state, action)
        
        reward, done = self._get_reward(problem_instance, state, action, next_state, num_action)  
        
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
        for bridge in adaptations:
            if hasattr(self, '_transform_action'):
                bridge = self._transform_action(bridge)
            row = bridge // self.num_cols
            col = bridge % self.num_cols
            if desc[row][col] == 2 or desc[row][col] == 3:
                continue
            desc[row][col] = 1
        
        return desc