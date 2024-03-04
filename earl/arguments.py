import argparse

def parse_num_instances():
    parser = argparse.ArgumentParser(description='Initial argument parser.')
    
    parser.add_argument(
        '--num_instances',
        type=int, 
        default=50, 
        help='Number of instances to generate dynamically.'
        )
    
    args, remaining_argv = parser.parse_known_args()
    
    return args.num_instances, remaining_argv

def get_arguments(num_instances, remaining_argv):
    parser = argparse.ArgumentParser(
        description='Teach a multi-agent system to create its own context-dependent language.'
        )
    
    parser.add_argument(
        '--approach', 
        type=str, 
        default='commutative_q_table', 
        choices=['basic_q_table', 'commutative_q_table', 'hallucinated_q_table', 'basic_lfa', 'commutative_lfa', 'hallucinated_lfa', 'attention_neuron'],
        help='Choose which approach to use {default_val: %(default)s, choices: [%(choices)s]}'
        )
    
    instance_choices = [f'instance_{i}' for i in range(num_instances)]
    parser.add_argument(
        '--problem_instance', 
        type=str, 
        default='instance_0', 
        choices=instance_choices + ['columns', 'pathway'],
        help='Which problem to attempt (only applies to big problem size) {default_val: %(default)s, choices: [%(choices)s]}'
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
    
    args = parser.parse_args(remaining_argv)
        
    return args.approach, args.problem_instance, bool(args.random_state), args.reward_prediction_type