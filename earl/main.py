import numpy as np
import gymnasium as gym

from arguments import parse_num_instances, get_arguments

from agents.q_table import BasicQTable, HallucinatedQTable, CommutativeQTable


class EARL:
    def __init__(self, num_instances, problem_size, random_state, reward_prediction_type):
        map_name = '8x8' if problem_size == 'big' else '4x4'
        
        self.env = gym.make(
            id='FrozenLake-v1', 
            map_name=map_name, 
            render_mode=None
            )
        
        rng = np.random.default_rng(seed=42)
        self.num_cols = self.env.unwrapped.ncol
        
        self.basic_q_table = BasicQTable(self.env, num_instances, random_state, reward_prediction_type, rng)
        self.commutative_q_table = CommutativeQTable(self.env, num_instances, random_state, reward_prediction_type, rng)
        self.hallucinated_q_table = HallucinatedQTable(self.env, num_instances, random_state, rng)
                
    def retrieve_modifications(self, approach, problem):            
        approach = getattr(self, approach)
        modification = approach.get_adaptations(problem)
            
        return modification
    

if __name__ == '__main__':
    num_instances, remaining_argv = parse_num_instances()
    approach, problem_instance, random_state, reward_prediction_type = get_arguments(num_instances, remaining_argv)
    problem_size = 'big' if problem_instance in ['columns', 'pathway'] else 'small'
    
    earl = EARL(num_instances, problem_size, random_state, reward_prediction_type)    
    
    modification_set = earl.retrieve_modifications(approach, problem_instance)