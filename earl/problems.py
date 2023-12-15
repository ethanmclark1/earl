import copy
import numpy as np

desc = {
    '4x4': [
        "FFFF",
        "FFFF",
        "FFFF",
        "FFFF",
        ],
    '8x8': [
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        ]
}

problems = {
    '4x4': {
        'instance_0': {
            'start_and_goal': [
                (0,2),
                (3,2)                
                ],
            'obstacles': [(1,1),(2,1),
                          (1,2),(2,2),
                          (2,3)]
        },
        'instance_1': {
            'start_and_goal': [
                (2,0),
                (1,3)
                ],
            'obstacles': [(1,1),(2,2),
                          (1,2),(2,1),
                          (3,3)]
        }
    },
    '8x8': {
        'instance_2': {
            'start_and_goal': [
                (3,0),
                (3,7)
            ],
            'obstacles': [(1,1),(2,1),(3,1),(4,1),(5,1),(6,1),
                          (1,3),(2,3),(3,3),(4,3),(5,3),(6,3),
                          (1,5),(2,5),(3,5),(4,5),(5,5),(6,5),
                          (0,6)]
            
        },
        'instance_3': {
            'start_and_goal': [
                (0,7),
                (7,7)
            ],
            'obstacles': [(1,6),(2,6),(3,6),(4,6),(5,6),(6,6),
                          (1,7),(2,7),(3,7),(4,7),(5,7),(6,7)]
        },
        'instance_4': {
            'start_and_goal': [
                (0,0),
                (0,7)
            ],
            'obstacles': [(0,1),(1,1),(2,1),(3,1),(4,1),(5,1),(6,1),
                          (0,2),(1,2),(2,2),(3,2),(4,2),(5,2),(6,2),
                          (0,3),(1,3),(2,3),(3,3),(4,3),(5,3),(6,3),
                          (0,4),(1,4),(2,4),(3,4),(4,4),(5,4),(6,4)]
        }
    }
}


def get_problem_list(grid_size):
    return list(problems[grid_size].keys())

def get_entity_positions(problem_instance, rng, grid_size, percent_obstacles):
    obstacle_constr = problems[grid_size][problem_instance]['obstacles']
    num_obstacles = int(len(obstacle_constr) * percent_obstacles)
    
    tmp_obs_constr = np.array(obstacle_constr)
    obs_idx = rng.choice(range(len(obstacle_constr)), size=num_obstacles, replace=False)
    obstacles = tmp_obs_constr[obs_idx]
    
    start, goal = problems[grid_size][problem_instance]['start_and_goal']
    
    return start, goal, obstacles
    
def get_instantiated_desc(problem_instance, grid_size, percent_obstacles):
    instance_desc = copy.deepcopy(desc[grid_size])
    
    start, goal, obstacles = get_entity_positions(problem_instance, grid_size, percent_obstacles)
        
    tmp_lst = list(instance_desc[start[0]])
    tmp_lst[start[1]] = 'S'
    instance_desc[start[0]] = "".join(tmp_lst)
    
    tmp_lst = list(instance_desc[goal[0]])
    tmp_lst[goal[1]] = 'G'
    instance_desc[goal[0]] = "".join(tmp_lst)
    
    for obstacle in obstacles:
        tmp_lst = list(instance_desc[obstacle[0]])
        tmp_lst[obstacle[1]] = 'H'
        instance_desc[obstacle[0]] = "".join(tmp_lst)
    
    list_of_lists = [list(row) for row in instance_desc]
    converted_desc = np.array(list_of_lists, dtype='|S1')
    
    return converted_desc