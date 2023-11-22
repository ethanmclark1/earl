import os
import copy
import wandb
import pickle
import problems
import numpy as np
import networkx as nx


class EA:
    def __init__(self, env, grid_size, num_obstacles):
        self._init_hyperparams()
        
        self.env = env
        self.num_cols = env.unwrapped.ncol
        self.num_obstacles = num_obstacles
        
        self.grid_size = grid_size
        self.grid_dims = env.unwrapped.desc.shape
        self.state_dims = env.observation_space.n
        # Add in dummy action to terminate the episode
        self.action_dims = env.observation_space.n + 1
        
        self.rng = np.random.default_rng(seed=42)
        
    def _init_hyperparams(self):
        self.sma_window = -50
        self.action_cost = -0.25
        self.configs_to_consider = 10
        self.action_success_rate = 0.75

    def _save(self, approach, problem_instance, adaptation):
        problem_instance = 'cheese'
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
    
    # Populate buffer with dummy episodes
    def _populate_buffer(self, problem_instance, start_state):
        name = self.__class__.__name__
        hallucinate = name == 'BDQN' or name == 'TD3'

        for _ in range(self.dummy_episodes):
            done = False
            num_action = 0
            action_seq = []
            state = start_state
            while not done:
                num_action += 1
                action = self.rng.integers(self.action_dims)
                reward, next_state, done = self._step(problem_instance, state, action, num_action)
                
                # Add action to action sequence to hallucinate traces if using DiscreteRL
                if hallucinate:
                    action_seq += [action]
                else:
                    self.buffer.add(state, action, reward, next_state, done)
                state = next_state
            
            if hallucinate:
                self.buffer.hallucinate(action_seq, reward)
    
    # Calculate rewards for a given configuration by averaging A* path lengths over multiple trials
    """
    Cell Values:
        0: Safe
        1: Bridge
        2: Start
        3: Goal
        4: Hole
    """
    def _get_reward(self, problem_instance, configuration):
        def manhattan_dist(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        rewards = []
        desc = copy.deepcopy(configuration).reshape(self.grid_dims)
        row_idx, col_idx = np.where(desc == 1)
        bridges = set(zip(row_idx, col_idx))
        
        for _ in range(self.configs_to_consider):
            graph = nx.grid_graph(dim=[self.num_cols, self.num_cols])
            start, goal, obstacles = problems.get_entity_positions(problem_instance, self.grid_size, self.num_obstacles)
            
            # Bridges cannot cover start or goal cells 
            if desc[start] == 1 or desc[goal] == 1:
                rewards += [-200]
                continue
            
            desc[start], desc[goal] = 2, 3
            # Populate obstacle in cell only if bridge is not already there
            for obstacle in obstacles:
                if desc[obstacle[0], obstacle[1]] != 1:
                    desc[obstacle[0], obstacle[1]] = 4
            
            for i in range(self.num_cols):
                for j in range(self.num_cols):
                    cell_value = desc[i, j]
                    for neighbor in graph.neighbors((i, j)):
                        weight = 1
                        if cell_value == 4:
                            weight = 1000
                        elif cell_value == 1:
                            weight = 0
                        graph[(i, j)][neighbor]['weight'] = weight
                        
            path = set(nx.astar_path(graph, start, goal, manhattan_dist, 'weight'))
            rewards += [-len(path - bridges)]

        avg_reward = np.mean(rewards)
        return avg_reward
    
    # Apply adaptation to task environment
    def _step(self, problem_instance, state, action, num_action):
        reward = 0
        state = copy.deepcopy(state)
        terminating_action = len(state)
        
        next_state = state
        # Add stochasticity to instrumental action execution
        if action != terminating_action and self.action_success_rate > self.rng.random():
            next_state[action] = 1
        
        done = action == terminating_action
        if done:
            reward = self.action_cost * num_action
            reward += self._get_reward(problem_instance, next_state)  
        
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
            print('Generating new adaptation...')
            adaptations, losses, rewards = self._generate_adaptations(problem_instance)
            self._save(approach, problem_instance, adaptations)
        
        print(f'{approach} adaptations for {problem_instance.capitalize()} problem instance:\n{adaptations}\n')
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