import problems
import numpy as np
import networkx as nx
import gymnasium as gym

from plotter import plot_metrics
from arguments import get_arguments

from agents.q_table import BasicQTable, HallucinatedQTable, CommutativeQTable
from agents.lfa import BasicLFA, HallucinatedLFA, CommutativeLFA
from agents.attention_neuron import AttentionNeuron


class Driver:
    def __init__(self, grid_size, has_max_action, render_mode):
        self.env = gym.make(
            id='FrozenLake-v1', 
            map_name=grid_size, 
            render_mode=render_mode
            )
        
        random_seed = 42
        self.grid_size = grid_size
        self.num_cols = self.env.unwrapped.ncol
        self.rng = np.random.default_rng(seed=random_seed)
        
        # Q-Learning
        self.basic_q_table = BasicQTable(self.env, has_max_action, self.rng)
        self.hallucinated_q_table = HallucinatedQTable(self.env, has_max_action, self.rng)
        self.commutative_q_table = CommutativeQTable(self.env, has_max_action, self.rng)
        
        # Function Approximations
        self.basic_lfa = BasicLFA(self.env, has_max_action, self.rng)
        self.hallucinated_lfa = HallucinatedLFA(self.env, has_max_action, self.rng)
        self.commutative_lfa = CommutativeLFA(self.env, has_max_action, self.rng)
        self.attention_neuron = AttentionNeuron(self.env, has_max_action, self.rng)
                
    def retrieve_modifications(self, problem_instance):
        approaches = [
            'basic_q_table',
            'hallucinated_q_table',
            'commutative_q_table',
            'basic_lfa',
            'hallucinated_lfa',
            'commutative_lfa',
            'attention_neuron'
        ]
        modification_set = {approach: None for approach in approaches}
        
        for name in approaches:
            approach = getattr(self, name)
            modification_set[name] = approach.get_adaptations(problem_instance)
            
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
        bridges = set(zip(bridge_row, bridge_col))
        
        return graph, start, goal, bridges
    
    def act(self, problem_instance, modification_set, num_episodes):
        approaches = modification_set.keys()
        path_len = {approach: [] for approach in approaches}
        avg_path_cost = {approach: 0 for approach in approaches}
        
        for _ in range(num_episodes):
            desc = problems.get_instantiated_desc(problem_instance, self.rng, self.grid_size, self.percent_obstacles)
            
            adapted_env = {}
            for approach, agent in approaches.items():
                tmp_desc = desc.copy()
                adapted_env[approach] = agent.get_adapted_env(tmp_desc, modification_set[approach])
                
            for approach, current_desc in adapted_env.items():
                graph, start, goal, bridges = self._make_graph(current_desc)
                path = nx.astar_path(graph, start, goal)
                path_len[approach] += [len(path) - len(bridges)]
                path = set(nx.astar_path(graph, start, goal))
            
        for approach in approaches:
            avg_path_cost[approach] = np.mean(path_len[approach])
        
        return avg_path_cost
    

if __name__ == '__main__':
    grid_size, has_max_action, render_mode = get_arguments()
    driver = Driver(grid_size, has_max_action, render_mode)    
    
    metric = []
    num_episodes = 10000
    problem_list = problems.get_problem_list(driver.grid_size)
    for problem_instance in problem_list:
        modification_set = driver.retrieve_modifications(problem_instance)
        avg_path_cost = driver.act(problem_instance, modification_set, num_episodes)
        metric.append(avg_path_cost)

    plot_metrics(problem_list, metric)