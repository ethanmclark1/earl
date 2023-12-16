import argparse

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Teach a multi-agent system to create its own context-dependent language.'
        )
            
    parser.add_argument(
        '--render_mode', 
        type=str, 
        default='human', 
        choices=['human', 'ansi', 'rgb_array', 'None'], 
        help='Mode of visualization {default_val: None, choices: [%(choices)s]}'
        )
    
    args = parser.parse_args()
        
    return args.render_mode