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
    'citycenter': {
        'starts': [(0,0),(0,1),(0,4),(0,7),(1,0),(1,1),(1,3),(1,7),(2,0),(2,3),(3,0),(3,2),(3,5),(4,0),(5,4),(7,1),(7,2),(7,4)],
        'goals': [(0,5),(2,4),(3,3),(3,4),(4,2),(4,3),(4,4),(4,5),(5,5),(7,0)],
        'holes': [            (0,2),(0,3),            (0,6),
                              (1,2),      (1,4),
                        (2,1),(2,2),            (2,5),(2,6),
                        (3,1),                        (3,6),
                                                      (4,6),
                        (5,1),(5,2),(5,3),            (5,6),
                        (6,1),(6,2),(6,3),(6,4),              
                                    (7,3),            (7,6),(7,7)]
    },
    'pathway': {
        'starts': [(0,0),(0,1),(0,2),(0,5),(0,6),(0,7),(1,6),(1,7),(2,7),(5,7),(6,6),(6,7),(7,0),(7,1),(7,5),(7,6),(7,7)],
        'goals': [(1,0),(1,3),(2,0),(2,1),(2,2),(3,0),(3,7),(4,0),(4,1),(4,2),(4,7),(5,1),(6,2),(7,3)],
        'holes': [     (1,1),(1,2),      (1,4),(1,5),
                                   (2,3),      (2,5),(2,6),
                       (3,1),(3,2),(3,3),(3,4),(3,5),(3,6),
                                               (4,5),(4,6),
                  (5,0),     (5,2),      (5,4),(5,5),(5,6),
                  (6,0),(6,1),           (6,4),(6,5),
                                         (7,4)]
    }
}


def get_problem_list():
    return list(problems.keys())

def get_entity_positions(problem_instance, rng, percent_holes):
    starts = problems[problem_instance]['starts']
    goals = problems[problem_instance]['goals']
    start = rng.choice(starts)
    goal = rng.choice(goals)

    holes = problems[problem_instance]['holes']
    num_holes = int(len(holes) * percent_holes)
    holes = rng.choice(holes, size=num_holes, replace=False)
    
    return tuple(start), tuple(goal), holes