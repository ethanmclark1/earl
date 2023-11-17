# This code is based on the following repository:
# https://github.com/JakeForsey/attention-neuron
# # Author: Jake Forsey (JakeForsey)
# Title: attention_neuron.ipynb
# Version: b4fec3b

import cma
import copy
import torch
import wandb
import numpy as np

from multiprocessing import Pool

from agents.utils.ea import EA
from agents.utils.network import PINN

# TODO: Validate everything
class AttentionNeuron(EA):
    def __init__(self, env, grid_size, num_obstacles):
        super(AttentionNeuron, self).__init__(env, grid_size, num_obstacles)
        
        self.pop_size = 50
        self.n_processes = 4
        self.n_generations = 300
        self.fitness_samples = 10
        self.attention_neuron = None
        
    def _init_wandb(self, problem_instance, affinity_instance):
        config = super()._init_wandb(problem_instance, affinity_instance)
        config.action_cost = self.action_cost
        config.configs_to_consider = self.configs_to_consider
        
    def _select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            action = self.attention_neuron(state)            
        return action
    
    # Set the parameters of the model
    def _set_params(self, model, parameters):
        i = 0
        parameters = torch.from_numpy(parameters).float()
        for param in model.parameters():
            param_size = param.numel()
            param_shape = param.data.shape        
        
            if len(param_shape) > 1:
                param.data = parameters[i: i + param_size].reshape(param_shape)
            else:
                param.data = parameters[i: i + param_size]

            i += param_size
    
    def _calc_fitness(self, model, problem_instance, affinity_instance, start_state):
        total_fitness = 0
        for _ in range(self.fitness_samples):
            done = False
            model.reset()
            num_action = 0
            state = start_state
            
            while not done:
                num_action += 1
                action = self._select_action(state)
                reward, next_state, done = self._step(problem_instance, affinity_instance, state, action, num_action)
                state = next_state
                
            total_fitness += reward
        
        avg_fitness = total_fitness / self.fitness_samples
        wandb.log({'Average Reward': avg_fitness})
        return avg_fitness      
    
    def _fit_model(self, problem_instance, affinity_instance, start_state):
        fitness_history = []
        
        best_params = None
        best_fitness = -np.inf
        
        pool = Pool(self.n_processes)
        models = [copy.deepcopy(self.attention_neuron) for _ in range(self.pop_size)]
        model_params = sum(param.numel() for param in self.attention_neuron.parameters())
        solver = cma.CMAEvolutionStrategy(
            x0=np.zeros(model_params), 
            sigma0=1.0, 
            inopts={'popsize': self.pop_size, 'randn': np.random.randn}
            )
        
        for _ in range(self.n_generations):
            state = start_state
            # Get initial population (parameters)
            pop_params = solver.ask()
            
            for model, params in zip(models, pop_params):
                self._set_params(model, params)
            
            args = [(model, problem_instance, affinity_instance, state) for model in models]
            self._calc_fitness(models[0], problem_instance, affinity_instance, state)
            pop_fitness = pool.starmap(self._calc_fitness, args)   
            fitness_history.append(pop_fitness)
            
            solver.tell(pop_params, [-i for i in pop_fitness])
            
            max_pop_fitness = max(pop_fitness)            
            if max_pop_fitness > best_fitness:
                best_fitness = max_pop_fitness
                best_params = pop_params[pop_fitness.index(max_pop_fitness)]
                
        return best_params, best_fitness, np.array(fitness_history)
            
    # Generate optimal adaptation for a given problem instance
    def _generate_adaptations(self, problem_instance, affinity_instance):
        # self._init_wandb(problem_instance, affinity_instance)
        
        self.attention_neuron = PINN(self.action_dims)
        
        start_state = np.array([0] * self.state_dims)
        best_params, best_fitness, fitness_history = self._fit_model(problem_instance, affinity_instance, start_state)
        best_actions = self._get_action_seq(problem_instance, affinity_instance, best_params, start_state)
        
        wandb.log({'Final Reward': best_fitness})
        wandb.log({'Final Actions': best_actions})
        wandb.finish()
        
        return best_actions