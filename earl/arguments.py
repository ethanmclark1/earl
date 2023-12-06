import argparse

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Teach a multi-agent system to create its own context-dependent language.'
        )
    
    # --problem_space determines the size of the grid & the number of possible actions
    # 'small' = 4x4 grid with 4 possible actions (center 2x2 square)
    # 'medium' = 4x4 grid with 16 possible actions
    # 'large' = 8x8 grid with 64 possible actions
    parser.add_argument(
        '--problem_space',
        type=str, 
        default='small',
        choices=['small', 'medium', 'large'],
        help='Size of the problem space {default_val: small, choices: [%(choices)s]}'
        )
    
    parser.add_argument(
        '--render_mode', 
        type=str, 
        default='human', 
        choices=['human', 'ansi', 'rgb_array'], 
        help='Mode of visualization {default_val: None, choices: [%(choices)s]}'
        )
    
    args = parser.parse_args()
        
    return args.problem_space, args.render_mode