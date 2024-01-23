import argparse

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Teach a multi-agent system to create its own context-dependent language.'
        )
    
    parser.add_argument(
        '--is_online',
        type=int,
        default=1,
        choices=[0, 1],
        help='Train the agent online or offline {default_val: %(default)s, choices: [%(choices)s]}'
        )
    
    parser.add_argument(
        '--approach', 
        type=str, 
        default='commutative_q_table', 
        choices=['basic_q_table', 'commutative_q_table', 'hallucinated_q_table', 'basic_lfa', 'commutative_lfa', 'hallucinated_lfa', 'attention_neuron'],
        help='Choose which approach to use {default_val: %(default)s, choices: [%(choices)s]}'
        )
    
    parser.add_argument(
        '--problem_instance', 
        type=str, 
        default='minefield', 
        choices=['minefield', 'neighbors'],
        help='Which problem to attempt {default_val: %(default)s, choices: [%(choices)s]}'
        )
            
    parser.add_argument(
        '--random_state', 
        type=int, 
        default=1, 
        choices=[0, 1], 
        help='Generate a random initial state for the agent {default_val: None, choices: [%(choices)s]}'
        )
    
    parser.add_argument(
        '--reward_prediction_type', 
        type=str, 
        default='approximate', 
        choices=['lookup', 'approximate'], 
        help='Type of way to predict the reward r_3 {default_val: %(default)s}'
        )
    
    args = parser.parse_args()
        
    return bool(args.is_online), args.approach, args.problem_instance, bool(args.random_state), args.reward_prediction_type