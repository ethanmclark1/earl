import copy
import torch
import wandb
import numpy as np

from agents.utils.ea import EA

class ContinuousRL(EA):
    def __init__(self, env, grid_size, num_obstacles):
        super(ContinuousRL, self).__init__(env, grid_size, num_obstacles)
        
    def _init_wandb(self, problem_instance):
        config = super()._init_wandb(problem_instance)
        
    def _select_action(self, state):
        a=3
        
    def _learn(self):
        a=3
    
    def _train(self, problem_instance, start_state):
        a=3
    
    def _generate_adaptations(self, problem_instance):
        a=3
        
    
        
    