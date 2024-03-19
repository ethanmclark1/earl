import yaml
import numpy as np
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

output_file = 'earl/problems.yaml'

def generate_problems(problem_size, rng, num_instances, action_size):
    if problem_size == '8x8':
        return
    
    with open(output_file, 'r') as file:
        problem_instances = yaml.safe_load(file) or {}
    
    data = problem_instances.get(problem_size, {})
    if data and len(data.keys()) == num_instances:
        return
    elif data and len(data.keys()) > num_instances:
        for i in range(num_instances, len(data.keys())):
            data.pop(f'instance_{i}')
    else:
        problems = {}
        grid_size = [4, 4]
        i = len(data.keys()) if data else 0
        
        while i < num_instances:
            start, goal, holes, mapping = generate_4x4_problem(rng, grid_size, action_size)
            
            G = nx.grid_2d_graph(*grid_size)
            for hole in holes:
                G.remove_node(tuple(hole))
            
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

def generate_4x4_problem(rng, grid_size, num_bridges):
    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    num_starts = rng.integers(1, 3)
    num_goals = rng.integers(3, 5)
    num_holes = rng.integers(3, 5)
    
    all_positions = [[i, j] for i in range(grid_size[0]) for j in range(grid_size[1])]
    residential_district = [pos for pos in all_positions if pos[0] < 2 and pos[1] < 2]
    commercial_district = [pos for pos in all_positions if pos[0] > 2 or pos[1] > 2]
    middle_district = [pos for pos in all_positions if pos not in residential_district and pos not in commercial_district]
    
    # Select start positions, prioritizing the residential district
    num_residential_starts = rng.integers(1, min(num_starts, len(residential_district)) + 1)
    residential_starts = rng.choice(residential_district, size=num_residential_starts, replace=False).tolist()
    commercial_starts = rng.choice(commercial_district, size=num_starts - num_residential_starts, replace=False).tolist()
    starts = residential_starts + commercial_starts
    
    residential_district = [pos for pos in residential_district if pos not in starts]
    commercial_district = [pos for pos in commercial_district if pos not in starts]
    middle_district = [pos for pos in middle_district if pos not in starts]
    remaining_positions = [pos for pos in all_positions if pos not in starts]
    
    # Select goal positions, prioritizing the commercial district
    num_commercial_goals = rng.integers(2, min(num_goals - 1, len(commercial_district)) + 1)
    commercial_goals = rng.choice(commercial_district, size=num_commercial_goals, replace=False).tolist()
    residential_goals = rng.choice(residential_district, size=num_goals - num_commercial_goals, replace=False).tolist()
    goals = commercial_goals + residential_goals
    
    residential_district = [pos for pos in residential_district if pos not in goals]
    commercial_district = [pos for pos in commercial_district if pos not in goals]
    middle_district = [pos for pos in middle_district if pos not in goals]
    remaining_positions = [pos for pos in remaining_positions if pos not in goals]
    
    # Select hole positions, prioritizing the middle district and creating detours
    surrounding_holes = []
    num_middle_holes = rng.choice(range(2, num_holes + 1))
    num_surrounding_holes = num_holes - num_middle_holes
    middle_holes = rng.choice(middle_district, size=min(num_middle_holes, len(middle_district)), replace=False).tolist()
    remaining_positions = [pos for pos in remaining_positions if pos not in middle_holes]
    for _ in range(num_surrounding_holes):
        if remaining_positions:
            hole = rng.choice(remaining_positions).tolist()
            surrounding_holes.append(hole)
            remaining_positions.remove(hole)
            # Create detours by blocking adjacent positions
            adjacent_positions = [[hole[0]+i, hole[1]+j] for i in [-1, 0, 1] for j in [-1, 0, 1] if i != 0 or j != 0]
            adjacent_positions = [pos for pos in adjacent_positions if pos in remaining_positions]
            if adjacent_positions:
                detour_hole = rng.choice(adjacent_positions).tolist()
                surrounding_holes.append(detour_hole)
                remaining_positions.remove(detour_hole)
    holes = middle_holes + surrounding_holes

    # Select bridge positions based on their potential to improve the path
    bridge_candidates = remaining_positions + holes
    bridge_positions = []
    for _ in range(num_bridges):
        if bridge_candidates:
            pos_improvements = [(pos, sum(manhattan_distance(s, g) - (manhattan_distance(s, pos) + manhattan_distance(pos, g)) for s in starts for g in goals)) for pos in bridge_candidates]
            pos_improvements.sort(key=lambda x: x[1], reverse=True)
            probs = np.linspace(start=1, stop=0, num=len(pos_improvements))
            norm_probs = probs / np.sum(probs)
            chosen_index = rng.choice(range(len(pos_improvements)), p=norm_probs)
            chosen_pos = pos_improvements[chosen_index][0]
            bridge_positions.append(chosen_pos)
            bridge_candidates.remove(chosen_pos)

    bridge_positions = sorted(bridge_positions)
    bridges = {i: [int(pos[0]), int(pos[1])] for i, pos in enumerate(bridge_positions, start=1)}
        
    return starts, goals, holes, bridges

def get_instance(problem_instance=None):
    with open(output_file, 'r') as file:
        problems = yaml.safe_load(file)
    
    if 'instance' in problem_instance:
        instance = problems['4x4'][problem_instance]
    else:
        instance = problems['8x8'][problem_instance]
    
    return instance

def get_entity_positions(instance, rng):
    start = tuple(rng.choice(instance['starts']))
    goal = tuple(rng.choice(instance['goals']))
    holes = [tuple(hole) for hole in instance['holes']]
    
    return start, goal, holes