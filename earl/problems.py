import copy
import numpy as np

desc = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
]


problems = {
    # 'horizontal': {
    #     'start_and_goal': [(1,4),(6,4)],
    #     'obstacles': [(2,2),(3,2),(4,2),(5,2),
    #                   (1,3),(2,3),(3,3),(4,3),(5,3),(6,3),
    #                   (2,4),(3,4),(4,4),(5,4),
    #                   (0,5),(1,5),(2,5),(3,5),(4,5),(5,5),(6,5),(7,5)]
    # },
    # 'staggered': {
    #     'start_and_goal': [(4,1),(4,6)],
    #     'obstacles': [(1,2),(2,2),(3,2),(4,2),(5,2),(6,2),(7,2),
    #                   (0,4),(1,4),(2,4),(3,4),(4,4),(5,4),(6,4),
    #                   (0,5),(1,5),(2,5),(3,5),(4,5),(5,5),(6,5)]
    # },
    'wall': {
        'start_and_goal': [(1,5),(6,5)],
        'obstacles': [(1,3),(2,3),(3,3),(4,3),(5,3),(6,3),
                      (1,4),(2,4),(3,4),(4,4),(5,4),(6,4),
                      (2,5),(3,5),(4,5),(5,5),
                      (0,6),(1,6),(2,6),(3,6),(4,6),(5,6),(6,6),(7,6)]
    }
}


def get_problem_list():
    return list(problems.keys())

def get_entity_positions(problem_instance, rng, percent_obstacles):
    obstacle_constr = problems[problem_instance]['obstacles']
    num_obstacles = int(len(obstacle_constr) * percent_obstacles)
    
    tmp_obs_constr = np.array(obstacle_constr)
    obs_idx = rng.choice(range(len(obstacle_constr)), size=num_obstacles, replace=False)
    obstacles = tmp_obs_constr[obs_idx]
    
    start, goal = problems[problem_instance]['start_and_goal']
    
    return start, goal, obstacles
    
def get_instantiated_desc(problem_instance, rng, percent_obstacles):
    instance_desc = copy.deepcopy(desc)
    
    start, goal, obstacles = get_entity_positions(problem_instance, rng, percent_obstacles)
    
    s_x, s_y = start
    g_x, g_y = goal
    instance_desc[s_x][s_y] = 2
    instance_desc[g_x][g_y] = 3
    
    for obstacle in obstacles:
        o_x, o_y = obstacle
        instance_desc[o_x][o_y] = 4
    
    return instance_desc