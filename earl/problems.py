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
        'cross': [(0,1),(1,1),(2,1),(3,1),(1,0),(1,2),(1,3)],
        'gate': [(1,0),(1,1),(1,3),(2,0),(2,1),(2,3)],
        'snake': [(0,1),(0,2),(1,2),(1,3),(2,3),(0,2),(0,3),(1,4)],
        'diagonal': [(0,0),(1,1),(2,2),(2,1),(1,2),(0,3)]
    },
    '8x8': {
        'cross': [(0,3),(1,3),(2,3),(3,3),(4,3),(5,3),(6,3),(7,3),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7)],
        'gate': [(0,3),(1,3),(2,3),(3,3),(5,3),(6,3),(7,3),(0,4),(1,4),(2,4),(3,4),(5,4),(6,4),(7,4)],
        'snake': [(0,1),(1,1),(1,2),(1,3),(2,3),(2,4),(2,5),(3,5),(3,6),(7,6),(6,6),(6,5),(6,4),(5,4),(5,3),(5,2),(4,2),(4,1)],
        'diagonal': [(0,0),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(1,6),(2,5),(3,4),(4,3),(5,2),(6,1)],
    }
}

def get_problem_list(grid_size):
    return list(problems[grid_size].keys())

def get_entity_positions(problem_instance, grid_size, num_obstacles):
    obstacle_constr = problems[grid_size][problem_instance]
    
    tmp_obs_constr = np.array(obstacle_constr)
    obs_idx = np.random.choice(range(len(obstacle_constr)), size=num_obstacles, replace=False)
    obstacles = tmp_obs_constr[obs_idx]
    
    dims = 4 if grid_size == '4x4' else 8
    all_pos = set((x, y) for x in range(dims) for y in range(dims))
    available_pos = list(all_pos - set(obstacle_constr))

    # Randomly select start and goal from available positions
    start_idx, goal_idx = np.random.choice(range(len(available_pos)), size=2, replace=False)
    start = available_pos[start_idx]
    goal = available_pos[goal_idx]
    
    return start, goal, obstacles
    
def get_instantiated_desc(problem_instance, grid_size, num_obstacles):
    instance_desc = copy.deepcopy(desc[grid_size])
    
    start, goal, obstacles = get_entity_positions(problem_instance, grid_size, num_obstacles)
        
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