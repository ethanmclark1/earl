import problems
import numpy as np
import gymnasium as gym

from itertools import product
from agents.q_table import BasicQTable, HallucinatedQTable, CommutativeQTable
from agents.lfa import BasicLFA, HallucinatedLFA, CommutativeLFA
from agents.attention_neuron import AttentionNeuron


class EARL:
    def __init__(self):
        self.env = gym.make(
            id='FrozenLake-v1', 
            map_name='8x8', 
            render_mode=None
            )
        
        random_seed = 42
        self.num_cols = self.env.unwrapped.ncol
        self.rng = np.random.default_rng(seed=random_seed)
        
        # Q-Learning
        self.basic_q_table = BasicQTable(self.env, self.rng)
        self.commutative_q_table = CommutativeQTable(self.env, self.rng)
        self.hallucinated_q_table = HallucinatedQTable(self.env, self.rng)
        
        # Function Approximations
        self.basic_lfa = BasicLFA(self.env, self.rng)
        self.commutative_lfa = CommutativeLFA(self.env, self.rng)
        self.hallucinated_lfa = HallucinatedLFA(self.env, self.rng)
        self.attention_neuron = AttentionNeuron(self.env, self.rng)
                
    def retrieve_modifications(self, problem_list):
        approaches = [
            'basic_q_table',
            'commutative_q_table',
            'hallucinated_q_table',
            # 'basic_lfa',
            # 'commutative_lfa',
            # 'hallucinated_lfa',
            # 'attention_neuron'
        ]
        modification_set = {approach: None for approach in approaches}
        
        for problem_instance, name in product(problem_list, approaches):
            approach = getattr(self, name)
            modification_set[name] = approach.get_adaptations(problem_instance)
            
        return modification_set
    

if __name__ == '__main__':
    earl = EARL()    
    
    metric = []
    problem_list = problems.get_problem_list()
    modification_set = earl.retrieve_modifications(problem_list)