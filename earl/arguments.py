import argparse

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Teach a multi-agent system to create its own context-dependent language.'
        )
    
    parser.add_argument(
        '--grid_size',
        type=str,
        default='4x4',
        choices=['4x4', '8x8'],
        help='Size of the grid {default_val: 4x4, choices: [%(choices)s]}'
        )
            
    parser.add_argument(
        '--render_mode', 
        type=str, 
        default='human', 
        choices=['human', 'ansi', 'rgb_array', 'None'], 
        help='Mode of visualization {default_val: None, choices: [%(choices)s]}'
        )
    
    args = parser.parse_args()
        
    return args.grid_size, args.render_mode