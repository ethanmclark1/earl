import copy
import numpy as np

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

problems = {
    'cross': {
        'obstacles': [
            [(1,3),(2,3),(3,3),(4,3),(5,3),(6,3)],
            [(3,1),(3,2),(3,3),(3,4),(3,5),(3,6)],
            ],
        'houses': [
            (1,1),(1,2),(1,4),(1,6),(4,0),(4,2),(4,4),(4,6),(6,1),(6,6),(7,3)
            ]
    },
    'gate': {
        'obstacles': [
            [(0,3),(1,3),(2,3),(3,3),(5,3),(6,3),(7,3)],
            [(0,4),(1,4),(2,4),(3,4),(5,4),(6,4),(7,4)],
        ],
        'houses': [
            (0,2),(0,5),(1,0),(1,7),(2,1),(2,6),(3,2),(3,5),(5,0),(5,7),(6,1),(6,6),(7,2),(7,5)
        ]
    },
    'snake': {
        'obstacles': [
            [(0,1),(1,1),(1,2),(1,3),(2,3),(2,4),(2,5),(3,5),(3,6)],
            [(7,6),(6,6),(6,5),(6,4),(5,4),(5,3),(5,2),(4,2),(4,1)]
        ],
        'houses': [
            (0,0),(0,2),(0,5),(0,7),(2,2),(2,6),(4,3),(5,1),(5,5),(6,3),(7,0),(7,5),(7,7)
        ]
    },
    'diagonal': {
        'obstacles': [
            [(0,0),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6)],
            [(1,6),(2,5),(3,4),(4,3),(5,2),(6,1)]
        ],
        'houses':[
            (0,1),(0,7),(1,0),(2,1),(2,3),(2,4),(2,6),(5,1),(5,6),(5,7),(6,2),(6,5)
        ]
    }
}

def get_problem_list():
    return list(problems.keys())

def get_affinity_list():
    return ['nearest', 'furthest', 'random']

def get_affinity_distribution(affinity_instance, start, objective_constr):
    def euclid_dist(point1, point2):
        return ((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2) ** 0.5
    
    if affinity_instance != 'random':
        reversed_order = True if affinity_instance == 'furthest' else False
        sorted_coords = sorted(objective_constr, key=lambda x: euclid_dist(start, x), reverse=reversed_order)
        sorted_coords = sorted_coords[1:]
        n = len(sorted_coords)
        r = 0.75
        prob = [r**i for i in range(n)]
        total = sum(prob)
        prob = [p/total for p in prob] 
    else:
        sorted_coords = copy.copy(objective_constr)
        sorted_coords.remove(start)
        prob = [1/len(sorted_coords)] * len(sorted_coords)
    
    return sorted_coords, prob    

def get_entity_positions(problem_instance, affinity_instance, num_obstacles):
    obstacle_constr, objective_constr = problems[problem_instance].values()
    
    start_idx = np.random.choice(range(len(objective_constr)))
    start = objective_constr[start_idx]
    sorted_coords, affinity_distr = get_affinity_distribution(affinity_instance, start, objective_constr)
    goal_idx = np.random.choice(range(len(sorted_coords)), p=affinity_distr)
    goal = sorted_coords[goal_idx]
        
    obstacles = []
    count = num_obstacles
    while count > 0:
        row_idx = count % len(obstacle_constr)
        row_len = len(obstacle_constr[row_idx])
        col_idx = np.random.choice(range(row_len))
        obstacles += [obstacle_constr[row_idx][col_idx]]
        count -= 1
    
    return start, goal, obstacles
    
def get_instantiated_desc(problem_instance, affinity_instance, num_obstacles):
    instance_desc = copy.deepcopy(desc)
    
    start, goal, obstacles = get_entity_positions(problem_instance, affinity_instance, num_obstacles)
        
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