import os
import copy
import wandb
import pickle
import problems
import numpy as np
import networkx as nx


class EA:
    def __init__(self, env, num_instances, random_state):
        self.env = env
        self.random_state = random_state
        
        self.rng = np.random.default_rng(seed=42)
        self.problem_size = env.spec.kwargs['map_name']
        self.max_action = 10 if self.problem_size == '8x8' else 4
        self.state_dims = 16 if self.problem_size == '8x8' else 6
        self.action_cost = 0.05 if self.problem_size == '8x8' else 0.05
        # Add a dummy action (+1) to terminate the episode
        self.action_dims = self.state_dims + 1    
        
        problems.generate_problems(self.problem_size, self.rng, num_instances, self.state_dims)
        self._generate_start_state = self._generate_random_state if random_state else self._generate_fixed_state
        
        # Stochasticity Parameters
        self.configs_to_consider = 5
        self.action_success_rate = 0.65
        
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
            project=f'{self.problem_size}', 
            entity='ethanmclark1', 
            name=f'{self.__class__.__name__} w/ {reward_prediction_type.capitalize()}',
            tags=[f'{problem_instance.capitalize()}', f'{self.problem_size}'],
            )
        
        config = wandb.config
        return config
    
    # Initialize action mapping for a given problem instance
    def _init_instance(self, problem_instance):   
        self.instance = problems.get_instance(problem_instance)
        
        warmup_episode_values = {'columns': 7500, 'pathway': 5000}
        self.warmup_episodes = warmup_episode_values.get(problem_instance, 0)

    def _generate_fixed_state(self):
        return np.zeros(self.grid_dims, dtype=int), []
    
    # Generate initial state for a given problem instance
    def _generate_random_state(self):
        bridge_list = list(self.instance['mapping'].values())
        
        num_bridges = self.rng.choice(self.max_action - 1)
        bridges = self.rng.choice(bridge_list, size=num_bridges, replace=False).tolist()
        state = np.zeros(self.grid_dims, dtype=int)
        
        for bridge in bridges:
            state[tuple(bridge)] = 1
            
        return state, bridges
    
    def _get_state_proxy(self, state):
        mutable_cells = self.instance['mapping'].values()
        state_proxy = state[[cell[0] for cell in mutable_cells], [cell[1] for cell in mutable_cells]]
        return state_proxy
    
    def _get_state_from_proxy(self, state_proxy):
        state = np.zeros(self.grid_dims, dtype=int)
        mutable_cells = list(self.instance['mapping'].values())
        
        for i, cell in enumerate(state_proxy):
            state[tuple(mutable_cells[i])] = cell
        
        return state
    
    # def _prepare_data(self, state_proxy, action, next_state_proxy):
    #     action_reshaped = np.array([action]).reshape(-1, 1)
    #     action_enc = self.encoder.fit_transform(action_reshaped).reshape(-1).astype(int)
        
    #     features = np.concatenate([state_proxy, action_enc, next_state_proxy])
    #     step_idx = self._get_index(features)
        
    #     return features.astype(float), step_idx
    
    def _reassign_states(self, prev_state_proxy, action, state_proxy, next_state_proxy):
        transformed_action = self._transform_action(action)
        
        prev_state = self._get_state_from_proxy(prev_state_proxy)
        commutative_state = self._place_bridge(prev_state, transformed_action)
        commutative_state_proxy = self._get_state_proxy(commutative_state)
        
        action_a_success = not np.array_equal(prev_state_proxy, state_proxy)
        action_b_success = not np.array_equal(state_proxy, next_state_proxy)
        
        if action_a_success and action_b_success:         
            pass   
        elif not action_a_success and action_b_success:
            next_state_proxy = commutative_state_proxy
        elif action_a_success and not action_b_success:
            commutative_state_proxy = prev_state_proxy
            next_state_proxy = state_proxy
        else:
            commutative_state_proxy = prev_state_proxy
            next_state_proxy = prev_state_proxy
            
        return commutative_state_proxy, next_state_proxy
                
    def _transform_action(self, action):
        if action == 0:
            return action

        return self.instance['mapping'][action]
    
    def _place_bridge(self, state, action):
        next_state = copy.deepcopy(state)
        
        if action != 0:      
            next_state[tuple(action)] = 1
            
        return next_state
    
    def _create_graph(self):
        graph = nx.grid_graph(dim=[self.num_cols, self.num_cols])
        nx.set_edge_attributes(graph, 100, 'weight')
        return graph     
        
    # Cell Values: {Frozen: 0, Bridge: 1, Start: 2, Goal: 3, Hole: 4}
    def _calc_utility(self, state):
        def manhattan_dist(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        utilities = []
        graph = self._create_graph()
        desc = copy.deepcopy(state).reshape(self.grid_dims)
        rows, cols = np.where(desc == 1)
        
        # Set up the bridges
        bridges = set(zip(rows, cols))
        for bridge in bridges:
            bridge = tuple(bridge)
            for neighbor in graph.neighbors(bridge):
                graph[bridge][neighbor]['weight'] = 0
        
        for _ in range(self.configs_to_consider):
            tmp_desc = copy.deepcopy(desc)
            tmp_graph = copy.deepcopy(graph)
            
            start, goal, obstacles = problems.get_entity_positions(self.instance, self.rng)
            tmp_desc[start], tmp_desc[goal] = 2, 3
            
            # Only place obstacles if the cell is frozen
            for obstacle in obstacles:
                obstacle = tuple(obstacle)
                if tmp_desc[obstacle] == 0:
                    tmp_desc[obstacle] = 4
                    tmp_graph.remove_node(obstacle)
            
            path = nx.astar_path(tmp_graph, start, goal, manhattan_dist, 'weight')
            utility = -len(set(path) - bridges)
            utilities.append(utility)

        avg_utility = np.mean(utilities)
        return avg_utility
    
    def _get_next_state(self, state, action):
        terminating_action = 0
        next_state = copy.deepcopy(state)
        transformed_action = self._transform_action(action)
        
        if transformed_action != terminating_action and self.action_success_rate >= self.rng.random():
            next_state = self._place_bridge(state, transformed_action)
            
        return next_state
    
    # r(s,a,s') = u(s') - u(s) - c(a)
    def _get_reward(self, state, action, next_state, num_action):
        reward = 0
        terminating_action = 0
        
        done = action == terminating_action
        timeout = num_action == self.max_action
        
        if not done:
            if not np.array_equal(state, next_state):
                util_s = self._calc_utility(state)
                util_s_prime = self._calc_utility(next_state)
                reward = util_s_prime - util_s
            reward -= self.action_cost * num_action
                        
        return reward, (done or timeout)
        
    # Apply adaptation to task environment
    def _step(self, state, action, num_action):
        next_state = self._get_next_state(state, action)
        reward, done = self._get_reward(state, action, next_state, num_action)  
        
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