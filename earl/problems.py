import yaml
import networkx as nx

desc = {
    '4x4': 
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ],
    '8x8':
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
}

output_file = 'earl/agents/utils/problems.yaml'

def generate_problems(problem_size, rng, num_instances):
    if problem_size == '8x8':
        return
    
    with open(output_file, 'r') as file:
        problem_instances = yaml.safe_load(file) or {}
        
    data = problem_instances.get(problem_size, {})
    if len(data.keys()) == num_instances:
        return
    elif len(data.keys()) > num_instances:
        for i in range(num_instances, len(data.keys())):
            data.pop(f'instance_{i}')
    else:    
        flattened_list = [y for x in desc[problem_size] for y in x]
        
        problems = {}
        i = len(data.keys())
        grid_size = (4, 4)
        num_starts = rng.integers(1, 3)
        num_goals = rng.integers(1, 5)
        num_holes = rng.integers(1, 6)
        all_positions = [[i, j] for i in range(grid_size[0]) for j in range(grid_size[1])]
        while i < num_instances:
            tmp_positions = all_positions.copy()
            entities_idx = rng.choice(len(flattened_list), size=num_starts+num_goals+num_holes, replace=False)
            entities = list(map(lambda x: [int(x // grid_size[0]), int(x % grid_size[1])], entities_idx))  
            start = entities[:num_starts]
            goal = entities[num_starts:num_starts+num_goals]
            holes = entities[num_starts+num_goals+1:]
            for pos in start + goal:
                tmp_positions.remove(pos)
            mapping_positions = rng.choice(len(tmp_positions), size=4, replace=False)
            mapping = {i: [int(pos // grid_size[0]), int(pos % grid_size[1])] for i, pos in enumerate(mapping_positions)}
            
            G = nx.grid_2d_graph(*grid_size)
            for hole in holes:
                hole = tuple(hole)
                G.remove_node(hole)
                
            if all(nx.has_path(G, tuple(s), tuple(g)) for s in start for g in goal):
                problem = {
                    'starts': start,
                    'goals': goal,
                    'holes': holes,
                    'mapping': mapping
                }
                problems[f'instance_{i}'] = problem
                i += 1

        problem_instances[problem_size] = problems

    with open(output_file, 'w') as file:
        yaml.dump(problem_instances, file)

def get_instance(problem_instance=None):
    with open(output_file, 'r') as file:
        problems = yaml.safe_load(file)
    
    if 'instance' in problem_instance:
        instance = problems['4x4'][problem_instance]
    else:
        instance = problems['8x8'][problem_instance]
    
    return instance

def get_entity_positions(instance, rng):
    starts = instance['starts']
    starts = [tuple(start) for start in starts]
    start = rng.choice(starts)
    goals = instance['goals']
    goals = [tuple(goal) for goal in goals]
    goal = rng.choice(goals)
    
    holes = instance['holes']
    
    return tuple(start), tuple(goal), holes