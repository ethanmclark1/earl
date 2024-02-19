import numpy as np
import gymnasium as gym

from arguments import get_arguments

from agents.q_table import BasicQTable, HallucinatedQTable, CommutativeQTable
from agents.lfa import BasicLFA, HallucinatedLFA, CommutativeLFA
from agents.attention_neuron import AttentionNeuron


class EARL:
    def __init__(self, random_state, reward_prediction_type):
        self.env = gym.make(
            id='FrozenLake-v1', 
            map_name='8x8', 
            render_mode=None
            )
        
        rng = np.random.default_rng(seed=42)
        self.num_cols = self.env.unwrapped.ncol
        
        # Q-Learning
        self.basic_q_table = BasicQTable(self.env, random_state, reward_prediction_type, rng)
        self.commutative_q_table = CommutativeQTable(self.env, random_state, reward_prediction_type, rng)
        self.hallucinated_q_table = HallucinatedQTable(self.env, random_state, rng)
        
        # Function Approximations
        self.basic_lfa = BasicLFA(self.env, random_state, reward_prediction_type, rng)
        self.commutative_lfa = CommutativeLFA(self.env, random_state, reward_prediction_type, rng)
        self.hallucinated_lfa = HallucinatedLFA(self.env, random_state, rng)
        self.attention_neuron = AttentionNeuron(self.env, random_state, rng)
                
    def retrieve_modifications(self, approach, problem):            
        approach = getattr(self, approach)
        modification = approach.get_adaptations(problem)
            
        return modification
    

if __name__ == '__main__':
    approach, problem_instance, random_state, reward_prediction_type = get_arguments()
    earl = EARL(random_state, reward_prediction_type)    
    
    modification_set = earl.retrieve_modifications(approach, problem_instance)