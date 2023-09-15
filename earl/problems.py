problems = {
    'corners': [
        [(0,1),(0,2),(1,0),(1,1),(2,0)],
        [(0,5),(0,6),(0,7),(1,6),(1,7),(2,7)],
        [(5,0),(6,0),(7,0),(6,1),(7,1),(7,2)],
        [(5,7),(6,6),(6,7),(6,6),(7,6),(7,7)],
    ]
}

def get_problem_list():
    return list(problems.keys())

def get_instance_constr(instance_name):
    instance_constr = problems[instance_name]
    return instance_constr

def get_desc(obstacles):
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
    
    for obstacle in obstacles:
        tmp_lst = list(desc[obstacle[0]])
        tmp_lst[obstacle[1]] = 'H'
        desc[obstacle[0]] = "".join(tmp_lst)
    
    return desc