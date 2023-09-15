import problems
import numpy as np
import gymnasium as gym

from agents.rl import RL
from agents.earl import EARL
from arguments import get_arguments


class Driver:
    def __init__(self, num_obstacles, render_mode):
        self.env = gym.make(
            id='FrozenLake-v1', 
            map_name='8x8', 
            render_mode=render_mode
            )
        
        self.earl = EARL(self.env, num_obstacles)
        self.num_obstacles = num_obstacles
    

        
    def act(self, problem_instance, train_epochs=20000, eval_epochs=100):
        reconfiguration = self.earl.get_reconfigured_env(problem_instance)
        desc = self._make_desc(problem_instance)
        self.env.unwrapped.desc = desc
        rec
        


if __name__ == '__main__':
    num_obstacles, render_mode = get_arguments()
    driver = Driver(num_obstacles, render_mode)    
    
    problem_list = problems.get_problem_list()
    for problem_instance in problem_list:
        driver.act(problem_instance)
    
    