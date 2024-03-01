import os
import copy
import wandb
import pickle
import problems
import numpy as np
import networkx as nx


class EA:
    def __init__(self, env, cost_fn, random_state, rng):
        self.env = env
        self.random_state = random_state
        self.rng = rng
        
        self.problem_size = env.spec.kwargs['map_name']
        # TODO: Generate 4x4 problems
        self._generate_init_state = self._generate_random_state if random_state else self._generate_fixed_state
        
        self.cost_fn = cost_fn
        self.action_cost = 0.05
        self.max_action = 10 if self.problem_size == '8x8' else 5
        self.state_dims = 16 if self.problem_size == '8x8' else 4
        # Add a dummy action (+1) to terminate the episode
        self.action_dims = self.state_dims + 1    
        
        # Stochasticity Parameters
        self.configs_to_consider = 1
        self.action_success_rate = 0.80
        
        self.num_cols = env.unwrapped.ncol
        self.grid_dims = env.unwrapped.desc.shape
        
        self.num_cols = env.unwrapped.ncol
        self.grid_dims = env.unwrapped.desc.shape

    def _save(self, approach, problem_instance, adaptation):
        directory = f'earl/agents/history/{approach.lower()}'
        filename = f'{self.problem_size}_{problem_instance}.pkl'
        file_path = os.path.join(directory, filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(file_path, 'wb') as file:
            pickle.dump(adaptation, file)
            
    def _load(self, approach, problem_instance):
        problem_instance = 'cheese'
        directory = f'earl/agents/history/{approach.lower()}'
        filename = f'{self.problem_size}_{problem_instance}.pkl'
        file_path = os.path.join(directory, filename)
        with open(file_path, 'rb') as f:
            adaptation = pickle.load(f)
        return adaptation
    
    def _init_wandb(self, problem_instance):
        if self.reward_prediction_type == 'approximate':
            reward_prediction_type = 'Estimator'
        else:
            reward_prediction_type = 'Lookup'
        
        wandb.init(
            project='earl', 
            entity='ethanmclark1', 
            name=f'{self.__class__.__name__} w/ {reward_prediction_type.capitalize()} & {self.cost_fn.capitalize()} Cost',
            tags=[f'{problem_instance.capitalize()}', f'{self.problem_size}'],
            )
        
        config = wandb.config
        return config
    
    # Initialize action mapping for a given problem instance
    def _init_mapping(self, problem_instance):   
        if self.problem_size == '8x8' and problem_instance == 'columns':
            self.mapping = {0: 9, 1: 14, 2: 19, 3: 27,
                            4: 28, 5: 29, 6: 30, 7: 35,
                            8: 36, 9: 41, 10: 43, 11: 44,
                            12: 56, 13: 57, 14: 59, 15: 63
                            }
            self.warmup_episodes = 15000
        elif self.problem_size == '8x8' and problem_instance == 'pathway':            
            self.mapping = {0: 9, 1: 13, 2: 25, 3: 26, 
                            4: 27, 5: 28, 6: 29, 7: 30, 
                            8: 41, 9: 43, 10: 45, 11: 47, 
                            12: 58, 13: 59, 14: 61, 15: 62
                            }
            self.warmup_episodes = 10000
    
    def _generate_fixed_state(self, problem_instance):
        return np.zeros(self.grid_dims, dtype=int), []
    
    # Generate initial state for a given problem instance
    def _generate_random_state(self, problem_instance):
        starts = problems.problems[self.problem_size][problem_instance]['starts']
        goals = problems.problems[self.problem_size][problem_instance]['goals']
        holes = problems.problems[self.problem_size][problem_instance]['holes']
        
        num_bridges = self.rng.choice(self.max_action)
        bridges = self.rng.choice(holes, size=num_bridges, replace=True)
        state = np.zeros(self.grid_dims, dtype=int)
        
        for bridge in bridges:
            bridge = tuple(bridge)
            if bridge in starts or bridge in goals:
                continue
            state[bridge] = 1
            
        return state, bridges
    
    def _get_mutable_state(self, state):
        mutable_cells = list(map(lambda x: (x // self.num_cols, x % self.num_cols), self.mapping.values()))
        rows, cols = zip(*mutable_cells)
        mutable_state = state[rows, cols]
        return mutable_state
        
    def _get_state_idx(self, state):
        mutable_state = self._get_mutable_state(state)
        binary_str = "".join(str(cell) for cell in reversed(mutable_state))
        state_idx = int(binary_str, 2)
        return state_idx   
    
    def _get_state_from_idx(self, state_idx):
        binary_str = format(state_idx, f'0{len(self.mapping)}b')[::-1]
        state = np.zeros((self.num_cols, self.num_cols), dtype=int)

        mutable_cells = list(map(lambda x: (x // self.num_cols, x % self.num_cols), self.mapping.values()))
        rows, cols = zip(*mutable_cells)

        for i, (row, col) in enumerate(zip(rows, cols)):
            state[row, col] = int(binary_str[i])

        return state

    def _transform_action(self, action):
        if action == self.action_dims - 1:
            return action

        return self.mapping[action]
    
    def _place_bridge(self, state, action):
        next_state = copy.deepcopy(state)
        row = action // self.num_cols
        col = action % self.num_cols
        bridge = (row, col)
        next_state[bridge] = 1
        return next_state
    
    def _create_graph(self):
        graph = nx.grid_graph(dim=[self.num_cols, self.num_cols])
        nx.set_edge_attributes(graph, 25, 'weight')
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
        
        # Set up the bridges
        bridges = set(zip(row_idx, col_idx))
        for bridge in bridges:
            bridge = tuple(bridge)
            for neighbor in graph.neighbors(bridge):
                graph[bridge][neighbor]['weight'] = 0
        
        for _ in range(self.configs_to_consider):
            tmp_desc = copy.deepcopy(desc)
            tmp_graph = copy.deepcopy(graph)
            start, goal, obstacles = problems.get_entity_positions(self.problem_size, problem_instance, self.rng)
            
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
    
    def _get_next_state(self, state, action):
        transformed_action = self._transform_action(action)
        next_state = copy.deepcopy(state)
        terminating_action = self.action_dims - 1
        # Add stochasticity to actions
        if transformed_action != terminating_action and self.action_success_rate >= self.rng.random():
            next_state = self._place_bridge(state, transformed_action)
            
        return next_state
    
    # r(s,a,s') = u(s') - u(s) - c(a)
    def _get_reward(self, problem_instance, state, action, next_state, num_action):
        reward = 0
        terminating_action = self.action_dims - 1
        
        done = action == terminating_action
        timeout = num_action == self.max_action
        
        if not done:
            # Check if state and next state are equal
            if not np.array_equal(state, next_state):
                util_s = self._calc_utility(problem_instance, state)
                util_s_prime = self._calc_utility(problem_instance, next_state)
                reward = util_s_prime - util_s
            
            if self.cost_fn == 'linear':
                action_cost = (self.action_cost * num_action)
            elif self.cost_fn == 'sublinear':
                action_cost = (self.action_cost * np.sqrt(num_action))
            elif self.cost_fn == 'logarithmic':
                action_cost = (self.action_cost * np.log(num_action))
            else:
                action_cost = self.action_cost
                
            reward -= action_cost
        
        return reward, (done or timeout)
        
    # Apply adaptation to task environment
    def _step(self, problem_instance, state, action, num_action):
        next_state = self._get_next_state(state, action)
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