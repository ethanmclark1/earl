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
    'cross': {
        'houses': [(4,1),(1,3),(6,3),(4,6)],
        'holes': [      (2,1),            (5,1),
                  (1,2),(2,2),(3,2),(4,2),(5,2),(6,2),
                        (2,3),(3,3),(4,3),(5,3),
                        (2,4),(3,4),(4,4),(5,4),
                  (1,5),(2,5),(3,5),(4,5),(5,5),(6,5),
                        (2,6),            (5,6)]
    },
    'twister': {
        'houses': [(4,1),(1,3),(6,4),(3,6)],
        'holes': [      (2,1),            (5,1),
                  (1,2),(2,2),(3,2),(4,2),(5,2),(6,2),
                        (2,3),(3,3),(4,3),(5,3),
                        (2,4),(3,4),(4,4),(5,4),
                  (1,5),(2,5),(3,5),(4,5),(5,5),(6,5),
                        (2,6),            (5,6)]
    }
}


def get_problem_list():
    return list(problems.keys())

def get_entity_positions(problem_instance, rng, percent_holes):
    houses = problems[problem_instance]['houses']
    houses = rng.choice(houses, size=2, replace=False)
    start, goal = houses
    
    holes = problems[problem_instance]['holes']
    num_holes = int(len(holes) * percent_holes)
    holes = rng.choice(holes, size=num_holes, replace=False)
    
    return tuple(start), tuple(goal), holes