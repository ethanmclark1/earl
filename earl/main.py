import copy
import problems
import numpy as np
import networkx as nx
import gymnasium as gym

from plotter import plot_metrics
from arguments import get_arguments

from agents.earl import EARL
from agents.discrete_rl import DiscreteRL
from agents.continuous_rl import ContinuousRL
from agents.attention_neuron import AttentionNeuron


class Driver:
    def __init__(self, grid_size, render_mode):
        self.env = gym.make(
            id='FrozenLake-v1', 
            map_name=grid_size, 
            render_mode=render_mode
            )
        
        num_obstacles = 5 if grid_size == '4x4' else 14
        
        self.grid_size = grid_size
        self.num_cols = self.env.unwrapped.ncol
        
        self.earl = EARL(self.env, grid_size, num_obstacles)
        self.discrete_rl = DiscreteRL(self.env, grid_size, num_obstacles)
        self.continuous_rl = ContinuousRL(self.env, grid_size, num_obstacles)
        self.attention_neuron = AttentionNeuron(self.env, grid_size, num_obstacles)
        
    def retrieve_modifications(self, problem_instance):
        # approaches = ['discrete_rl', 'continuous_rl', 'attention_neuron', 'earl']
        approaches = ['discrete_rl']
        losses = {"A* w/ DiscreteRL": None, "A* w/ ContinuousRL": None, "A* w/ AttentionNeuron": None, "A* w/ EARL": None}
        rewards = {"A* w/ DiscreteRL": None, "A* w/ ContinuousRL": None, "A* w/ AttentionNeuron": None, "A* w/ EARL": None}
        modification_set = {approach: None for approach in approaches}
        
        for idx, name in enumerate(approaches):
            modification_set[name], loss, reward = getattr(self, approaches[idx]).get_adaptations(problem_instance)
            losses[list(losses.keys())[idx]] = loss
            rewards[list(losses.keys())[idx]] = reward
            
        return modification_set, losses, rewards
    
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
        transporter_row, transporter_col = np.where(desc == b'T')
        transporter = set(zip(transporter_row, transporter_col))
        
        return graph, start, goal, transporter
    
    def act(self, problem_instance, modification_set, num_episodes):
        path_len = {'A* w/ EARL': [], 'A* w/ DiscreteRL': []}
        avg_path_cost = {'A* w/ EARL': 0, 'A* w/ DiscreteRL': 0}
        
        for _ in range(num_episodes):
            desc = problems.get_instantiated_desc(problem_instance, self.grid_size, self.num_obstacles)
            tmp_desc = copy.deepcopy(desc)
            earl_desc = self.earl.get_adapted_env(tmp_desc, modification_set['earl'])
            tmp_desc = copy.deepcopy(desc)
            discrete_rl_desc = self.discrete_rl.get_adapted_env(tmp_desc, modification_set['rl'])
            
            for approach in path_len.keys():
                current_desc = discrete_rl_desc if approach == 'A* w/ DiscreteRL' else earl_desc
                graph, start, goal, transporter = self._make_graph(current_desc)
                path = set(nx.astar_path(graph, start, goal))
                path_len[approach] += [len(path - transporter)]
                
        avg_path_cost['A* w/ EARL'] = np.mean(path_len['A* w/ EARL'])
        avg_path_cost['A* w/ DiscreteRL'] = np.mean(path_len['A* w/ DiscreteRL'])
        return avg_path_cost
    

if __name__ == '__main__':
    grid_size, render_mode = get_arguments()
    driver = Driver(grid_size, render_mode)    
    
    all_metrics = []
    num_episodes = 10000
    problem_list = problems.get_problem_list(grid_size)
    for problem_instance in problem_list:
        modification_set, losses, rewards = driver.retrieve_modifications(problem_instance)
        avg_path_cost = driver.act(problem_instance, modification_set, num_episodes)
        all_metrics.append({
            'avg_path_cost': avg_path_cost,
            'losses': losses,
            'rewards': rewards
        })
    
    # plot_metrics(problem_list, affinity_list, all_metrics)