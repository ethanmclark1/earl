import argparse

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Teach a multi-agent system to create its own context-dependent language.'
        )
    
    parser.add_argument(
        '--approach', 
        type=str, 
        default='commutative_q_table', 
        choices=['basic_q_table', 'commutative_q_table', 'hallucinated_q_table', 'basic_lfa', 'commutative_lfa', 'hallucinated_lfa', 'attention_neuron'],
        help='Choose which approach to use {default_val: basic_q_table, choices: [%(choices)s]}'
        )
    
    parser.add_argument(
        '--problem_instance', 
        type=str, 
        default='cross', 
        choices=['cross', 'twister'],
        help='Which problem to attempt {default_val: cross, choices: [%(choices)s]}'
        )
            
    parser.add_argument(
        '--random_state', 
        type=int, 
        default=0, 
        choices=[0, 1], 
        help='Generate a random initial state for the agent {default_val: None, choices: [%(choices)s]}'
        )
    
    parser.add_argument(
        '--reward_prediction_type', 
        type=str, 
        default='linear', 
        choices=['table', 'median_table', 'mean_table', 'linear', 'noisy_linear', 'median_noisy_linear', 'mean_noisy_linear', 'noisiest_linear', 'median_noisiest_linear', 'mean_noisiest_linear'], 
        help='Type of way to predict the reward r_3 {default_val: %(default)s}'
        )
    
    args = parser.parse_args()
        
    return args.approach, args.problem_instance, bool(args.random_state), args.reward_prediction_type