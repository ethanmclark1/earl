import copy
import problems
import numpy as np
import networkx as nx
import gymnasium as gym

from plotter import plot_metrics
from arguments import get_arguments

from agents.td3 import TD3
from agents.earl import EARL
from agents.bdqn import BDQN
from agents.attention_neuron import AttentionNeuron


class Driver:
    def __init__(self, grid_size, render_mode):
        self.env = gym.make(
            id='FrozenLake-v1', 
            map_name=grid_size, 
            render_mode=render_mode
            )
        
        
        self.grid_size = grid_size
        self.num_cols = self.env.unwrapped.ncol
        self.num_obstacles = 5 if grid_size == '4x4' else 14
        
        self.earl = EARL(self.env, grid_size, self.num_obstacles)
        self.bdqn = BDQN(self.env, grid_size, self.num_obstacles)
        self.td3 = TD3(self.env, grid_size, self.num_obstacles)
        self.attention_neuron = AttentionNeuron(self.env, grid_size, self.num_obstacles)
        
    def retrieve_modifications(self, problem_instance):
        approaches = ['bdqn', 'td3', 'attention_neuron', 'earl']
        modification_set = {approach: None for approach in approaches}
        
        for idx, name in enumerate(approaches):
            modification_set[name] = getattr(self, approaches[idx]).get_adaptations(problem_instance)
            
        return modification_set
    
    # Make graph for A* search
    def _make_graph(self, desc):
        graph = nx.grid_graph(dim=(self.num_cols, self.num_cols))
        for i in range(self.num_cols):
            for j in range(self.num_cols):
                cell_value = desc[i, j]
                for neighbor in graph.neighbors((i, j)):
                    weight = 1
                    if cell_value == b'H':
                        weight = 1000
                    elif cell_value == b'T':
                        weight = 0
                    graph[(i, j)][neighbor]['weight'] = weight
        
        start_row, start_col = np.where(desc == b'S')
        start = list(zip(start_row, start_col))[0]
        goal_row, goal_col = np.where(desc == b'G')
        goal = list(zip(goal_row, goal_col))[0]
        bridge_row, bridge_col = np.where(desc == b'T')
        bridge = set(zip(bridge_row, bridge_col))
        
        return graph, start, goal, bridge
    
    def act(self, problem_instance, modification_set, num_episodes):
        path_len = {'A* w/ BDQN': [], 'A* w/ TD3': [], 'A* w/ AttentionNeuron': [], 'A* w/ EARL': []}
        avg_path_cost = {'A* w/ BDQN': 0, 'A* w/ TD3': 0, 'A* w/ AttentionNeuron': 0, 'A* w/ EARL': 0}
        
        for _ in range(num_episodes):
            desc = problems.get_instantiated_desc(problem_instance, self.grid_size, self.num_obstacles)
            
            # Create copies of desc for each approach due to pass-by-object-reference in Python
            tmp_desc = copy.deepcopy(desc)
            bdqn_desc = self.bdqn.get_adapted_env(tmp_desc, modification_set['bdqn'])
            tmp_desc = copy.deepcopy(desc)
            td3_desc = self.td3.get_adapted_env(tmp_desc, modification_set['td3'])
            tmp_desc = copy.deepcopy(desc)
            attention_neuron_desc = self.attention_neuron.get_adapted_env(tmp_desc, modification_set['attention_neuron'])
            tmp_desc = copy.deepcopy(desc)
            earl_desc = self.earl.get_adapted_env(tmp_desc, modification_set['earl'])
            
            for approach in path_len.keys():
                if approach == 'A* w/ BDQN':
                    current_desc = bdqn_desc
                elif approach == 'A* w/ TD3':
                    current_desc = td3_desc
                elif approach == 'A* w/ AttentionNeuron':
                    current_desc = attention_neuron_desc
                else:
                    current_desc = earl_desc
                    
                graph, start, goal, bridge = self._make_graph(current_desc)
                path = set(nx.astar_path(graph, start, goal))
                path_len[approach] += [len(path - bridge)]
        
        avg_path_cost['A* w/ BDQN'] = np.mean(path_len['A* w/ BDQN'])
        avg_path_cost['A* w/ TD3'] = np.mean(path_len['A* w/ TD3'])
        avg_path_cost['A* w/ AttentionNeuron'] = np.mean(path_len['A* w/ AttentionNeuron'])
        avg_path_cost['A* w/ EARL'] = np.mean(path_len['A* w/ EARL'])
        
        return avg_path_cost
    

if __name__ == '__main__':
    grid_size, render_mode = get_arguments()
    driver = Driver(grid_size, render_mode)    
    
    metric = []
    num_episodes = 10000
    problem_list = problems.get_problem_list(grid_size)
    for problem_instance in problem_list:
        modification_set = driver.retrieve_modifications(problem_instance)
        avg_path_cost = driver.act(problem_instance, modification_set, num_episodes)
        metric.append(avg_path_cost)

    plot_metrics(problem_list, metric)