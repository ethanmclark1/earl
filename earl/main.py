import copy
import problems
import numpy as np
import networkx as nx
import gymnasium as gym

from agents.rl import RL
from agents.earl import EARL
from plotter import plot_metrics
from arguments import get_arguments


class Driver:
    def __init__(self, num_obstacles, render_mode):
        self.env = gym.make(
            id='FrozenLake-v1', 
            map_name='8x8', 
            render_mode=render_mode
            )
        
        self.num_cols = self.env.unwrapped.ncol
        self.num_obstacles = num_obstacles
        self.rl = RL(self.env, num_obstacles)
        self.earl = EARL(self.env, num_obstacles)
        
    def retrieve_modifications(self, problem_instance):
        approaches = ['rl', 'earl']
        modification_set = {approach: None for approach in approaches}
        
        for idx, name in enumerate(approaches):
            approach = getattr(self, name)
            if hasattr(approach, 'get_adaptations'):
                modification_set[name] = getattr(self, approaches[idx]).get_adaptations(problem_instance)
            
        return modification_set
    
    # Make graph for A* search
    def _make_graph(self, desc):
        graph = nx.grid_graph(dim=(self.num_cols, self.num_cols))
        for i in range(self.num_cols):
            for j in range(self.num_cols):
                cell_value = desc[i, j]
                for neighbor in graph.neighbors((i, j)):
                    weight = 6
                    if cell_value == b'H':
                        weight = 100
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
        path_len = {'A* w/ EARL': [], 'A* w/ RL': []}
        avg_path_len = {'A* w/ EARL': 0, 'A* w/ RL': 0}
        
        for _ in range(num_episodes):
            desc = problems.get_instantiated_desc(problem_instance, self.num_obstacles)
            tmp_desc = copy.deepcopy(desc)
            rl_desc = self.rl.get_adapted_env(tmp_desc, modification_set['rl'])
            earl_desc = self.earl.get_adapted_env(tmp_desc, modification_set['earl'])
            
            for approach in path_len.keys():
                current_desc = rl_desc if approach == 'A* w/ RL' else earl_desc
                graph, start, goal, transporter = self._make_graph(current_desc)
                path = set(nx.astar_path(graph, start, goal))
                path -= transporter
                path_len[approach] += [len(path)]
                
        avg_path_len['A*'] = np.mean(path_len['A*'])
        avg_path_len['A* w/ EARL'] = np.mean(path_len['A* w/ EARL'])
        return avg_path_len
    

if __name__ == '__main__':
    num_obstacles, render_mode = get_arguments()
    driver = Driver(num_obstacles, render_mode)    
    
    metrics = []
    num_episodes = 10000
    problem_list = problems.get_problem_list()
    for problem_instance in problem_list:
        modification_set = driver.retrieve_modifications(problem_instance)
        avg_path_len = driver.act(problem_instance, modification_set, num_episodes)
        metrics += [avg_path_len]
    
    plot_metrics(problem_list, metrics)