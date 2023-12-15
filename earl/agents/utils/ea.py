import os
import copy
import wandb
import pickle
import problems
import numpy as np
import networkx as nx


class EA:
    def __init__(self, env, rng):
        self.env = env
        self.rng = rng
        
        self.num_cols = env.unwrapped.ncol
        self.grid_dims = env.unwrapped.desc.shape
        self.state_dims = env.observation_space.n
        self.grid_size = '4x4' if self.grid_dims[0] == 4 else '8x8'
        
        self._init_hyperparams()
        
    def _init_hyperparams(self):
        if self.grid_size == '4x4':
            self.max_action = 6
            self.action_cost = 0.50
            self.configs_to_consider = 25
            self.percent_obstacles = 0.75
        else:
            self.max_action = 8
            self.action_cost = 0.20
            self.configs_to_consider = 50
            self.percent_obstacles = 0.90
            
        self.action_success_rate = 0.75        
        self.sma_percentage = 0.025

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
    
    # Generate initial adaptations for a given problem instance
    def _generate_state(self, problem_instance):
        start, goal = problems.problems[self.grid_size][problem_instance]['start_and_goal']
        num_bridges = self.rng.choice(self.max_action)
        bridges = self.rng.choice(self.action_dims - 1, size=num_bridges, replace=False)
        bridges = set(bridges) - set([start, goal])
        state = np.zeros(self.state_dims, dtype=int)
        for bridge in bridges:
            if hasattr(self, '_transform_action'):
                bridge = self._transform_action(bridge)
            tmp_next_state = state.reshape(self.grid_dims)
            col = bridge // self.num_cols
            row = bridge % self.num_cols
            tmp_next_state[row, col] = 1
            state = tmp_next_state.reshape(self.state_dims)
        return state
    
    def _create_graph(self):
        graph = nx.grid_graph(dim=[self.num_cols, self.num_cols])
        nx.set_edge_attributes(graph, 5, 'weight')
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
            start, goal, obstacles = problems.get_entity_positions(problem_instance, self.rng, self.grid_size, self.percent_obstacles)
            
            # Bridges cannot cover start or goal cells 
            if tmp_desc[start] == 1 or tmp_desc[goal] == 1:
                utilities += [-self.state_dims]
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
        
        # Non-terminating actions incur an additional cost per action
        if not done:
            util_s = self._calc_utility(problem_instance, state)
            util_s_prime = self._calc_utility(problem_instance, next_state)
            reward = util_s_prime - util_s - (self.action_cost * num_action)
        
        return reward, (done or timeout)
        
    # Apply adaptation to task environment
    def _step(self, problem_instance, state, action, num_action):
        next_state = copy.deepcopy(state)
        terminating_action = self.action_dims - 1
        
        # Add stochasticity to actions
        if action != terminating_action and self.action_success_rate > self.rng.random():
            tmp_next_state = next_state.reshape(self.grid_dims)
            col = action // self.num_cols
            row = action % self.num_cols
            tmp_next_state[row, col] = 1
            next_state = tmp_next_state.reshape(self.state_dims)
        
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
        for reconfigruation in adaptations:
            row = reconfigruation // self.num_cols
            col = reconfigruation % self.num_cols
            if desc[row, col] == b'S' or desc[row, col] == b'G':
                continue
            desc[row, col] = b'T'
        
        return desc