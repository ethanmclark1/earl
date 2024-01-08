import argparse

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Teach a multi-agent system to create its own context-dependent language.'
        )
            
    parser.add_argument(
        '--random_state', 
        type=bool, 
        default=False, 
        choices=[True, False], 
        help='Generate a random initial state for the agent {default_val: None, choices: [%(choices)s]}'
        )
    
    args = parser.parse_args()
        
    return args.random_state