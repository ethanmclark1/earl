import copy
import numpy as np

from itertools import chain

problems = {
    'cross': [
        [(1,3),(2,3),(3,3),(4,3),(5,3),(6,3)],
        [(3,1),(3,2),(3,3),(3,4),(3,5),(3,6)],
    ],
    'gate': [
        [(0,3),(1,3),(2,3),(3,3),(5,3),(6,3),(7,3)],
        [(0,4),(1,4),(2,4),(3,4),(5,4),(6,4),(7,4)],
    ],
    'snake': [
        [(0,1),(1,1),(1,2),(1,3),(2,3),(2,4),(2,5),(3,5),(3,6)],
        [(7,6),(6,6),(6,5),(6,4),(5,4),(5,3),(5,2),(4,2),(4,1)]
    ],
    'diagonal': [
        [(0,0),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6)],
        [(1,6),(2,5),(3,4),(4,3),(5,2),(6,1)]
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

def get_entity_positions(problem_instance, num_obstacles):
    instance_constr = problems[problem_instance]
    all_states = set((i,j) for i in range(8) for j in range(8))
    unavailable_states = set(chain.from_iterable(instance_constr))
    available_states = list(all_states - unavailable_states)
    available_states_len = len(available_states)
    start_idx, goal_idx = np.random.choice(range(available_states_len), size=2, replace=False)
    start = available_states[start_idx]
    goal = available_states[goal_idx]
    
    obstacles = []
    count = num_obstacles
    while count > 0:
        row_idx = count % len(instance_constr)
        row_len = len(instance_constr[row_idx])
        col_idx = np.random.choice(range(row_len))
        obstacles += [instance_constr[row_idx][col_idx]]
        count -= 1
    
    return start, goal, obstacles
    
def get_instantiated_desc(problem_instance, num_obstacles):
    instance_desc = copy.deepcopy(desc)
    
    start, goal, obstacles = get_entity_positions(problem_instance, num_obstacles)
        
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