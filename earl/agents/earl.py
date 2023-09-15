import os
import pickle
import problems
import numpy as np
import networkx as nx

class EARL:
    def __init__(self, env, num_obstacles):
        self._init_hyperparams()
                
        self.env = env
        state_dims = env.observation_space.n
        action_dims = env.action_space.n
        self.dims = env.ncols
        
        self.num_obstacles = num_obstacles
        self.q_table = np.zeros((state_dims, action_dims))
        
    def _save(problem_instance, reconfiguration):
        directory = 'earl/agents/history'
        filename = f'{problem_instance}'
        file_path = os.path.join(directory, filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(file_path, 'wb') as file:
            pickle.dump(reconfiguration, file)
            
    def _load(self, problem_instance):
        directory = 'earl/agents/history'
        filename = f'{problem_instance}.pkl'
        file_path = os.path.join(directory, filename)
        with open(file_path, 'rb') as f:
            reconfiguration = pickle.load(f)
        return reconfiguration
            
    def _init_hyperparams(self):
        self.alpha = 0.7
        self.epsilon = 1
        self.gamma = 0.95
        self.epsilon_decay = 0.9999
        
    def _get_action(self):
        if np.random.random() > self.epsilon:
            action = np.argmax(self.value_function[:])
        else:
            action = np.random.randint(len(self.value_function))
        return action
    
    # TODO: Generate start and goal positions as well
    def _generate_instance(self, problem_instance):
        obstacles = []
        count = self.num_obstacles
        
        instance_constr = problems.get_instance_constr(problem_instance)
        while count > 0:
            row_idx = count % len(instance_constr)
            row_len = len(instance_constr[row_idx])
            col_idx = np.random.choice(range(row_len))
            obstacles += [instance_constr[row_idx][col_idx]]
            count -= 1
        
        desc = problems.get_desc(obstacles)
        
        list_of_lists = [list(row) for row in desc]
        converted_desc = np.array(list_of_lists, dtype='|S1')
        
        return converted_desc
    
    def _generate_reconfiguration(self, problem_instance):
        desc = self._generate_instance(problem_instance)
        a=3
    
    def get_reconfigured_env(self, problem_instance):
        try:
            reconfiguration = self._load(problem_instance)
        except FileNotFoundError:
            print(f'No stored reconfiguration for {problem_instance} problem instance.')
            print('Generating new reconfiguration...\n')
            reconfiguration = self._generate_reconfiguration(problem_instance)
            self._save(problem_instance, reconfiguration)
        
        return reconfiguration