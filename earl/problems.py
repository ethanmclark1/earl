import copy
import numpy as np

from itertools import chain

problems = {
    'corners': [
        [(0,1),(0,2),(1,0),(1,1),(2,0)],
        [(0,5),(0,6),(0,7),(1,6),(1,7),(2,7)],
        [(5,0),(6,0),(7,0),(6,1),(7,1),(7,2)],
        [(5,7),(6,6),(6,7),(6,6),(7,6),(7,7)],
    ]
}

desc = [
    "FFFFFFFF",
    "FFFFFFFF",
    "FFFFFFFF",
    "FFFFFFFF",
    "FFFFFFFF",
    "FFFFFFFF",
    "FFFFFFFF",
    "FFFFFFFF",
]

def get_problem_list():
    return list(problems.keys())

def get_instance_constr(instance_name):
    instance_constr = problems[instance_name]
    return instance_constr

def get_instantiated_desc(problem_instance, num_obstacles):
    obstacles = []
    count = num_obstacles
    instance_desc = copy.deepcopy(desc)
    
    instance_constr = get_instance_constr(problem_instance)
    all_states = set((i,j) for i in range(8) for j in range(8))
    unavailable_states = set(chain.from_iterable(instance_constr))
    available_states = list(all_states - unavailable_states)
    available_states_len = len(available_states)
    start_idx, goal_idx = np.random.choice(range(available_states_len), size=2, replace=False)
    start = available_states[start_idx]
    goal = available_states[goal_idx]
    
    tmp_lst = list(instance_desc[start[0]])
    tmp_lst[start[1]] = 'S'
    instance_desc[start[0]] = "".join(tmp_lst)
    
    tmp_lst = list(instance_desc[goal[0]])
    tmp_lst[goal[1]] = 'G'
    instance_desc[goal[0]] = "".join(tmp_lst)
    
    while count > 0:
        row_idx = count % len(instance_constr)
        row_len = len(instance_constr[row_idx])
        col_idx = np.random.choice(range(row_len))
        obstacles += [instance_constr[row_idx][col_idx]]
        count -= 1
    
    for obstacle in obstacles:
        tmp_lst = list(instance_desc[obstacle[0]])
        tmp_lst[obstacle[1]] = 'H'
        instance_desc[obstacle[0]] = "".join(tmp_lst)
    
    list_of_lists = [list(row) for row in instance_desc]
    converted_desc = np.array(list_of_lists, dtype='|S1')
    
    return converted_desc