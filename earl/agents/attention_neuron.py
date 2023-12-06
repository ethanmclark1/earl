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
from agents.utils.networks import PINN


# CMA-ES with Attention Mechanism
class AttentionNeuron(EA):
    def __init__(self, env, grid_size, num_obstacles):
        super(AttentionNeuron, self).__init__(env, grid_size, num_obstacles)
        
        self.attention_neuron = None
        
        self.n_processes = 16
        self.n_population = 50
        self.n_generations = 1000
        self.fitness_samples = 50
        self.sma_window = int(self.n_generations * self.sma_percentage)
        
    def _init_wandb(self, problem_instance):
        config = super()._init_wandb(problem_instance)
        config.action_cost = self.action_cost
        config.n_processes = self.n_processes
        config.n_population = self.n_population
        config.n_generations = self.n_generations
        config.fitness_samples = self.fitness_samples
        config.action_success_rate = self.action_success_rate
        
    def _select_action(self, state):
        with torch.no_grad():
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
    
    def _calc_fitness(self, model, problem_instance, start_state):
        total_fitness = []
        for _ in range(self.fitness_samples):
            done = False
            model.reset()
            num_action = 0
            state = start_state
            
            while not done:
                num_action += 1
                action = self._select_action(state)
                fitness, next_state, done = self._step(problem_instance, state, action, num_action)
                state = next_state
                
            total_fitness += [fitness]
        
        avg_fitness = np.mean(total_fitness)
        return avg_fitness      
    
    def _fit_model(self, problem_instance):        
        fitnesses = []
        best_params = None
        best_fitness = -np.inf
        
        pool = Pool(self.n_processes)
        models = [copy.deepcopy(self.attention_neuron) for _ in range(self.n_population)]
        model_params = sum(param.numel() for param in self.attention_neuron.parameters())
        solver = cma.CMAEvolutionStrategy(
            x0=np.zeros(model_params), 
            sigma0=1.0, 
            inopts={'popsize': self.n_population, 'randn': np.random.randn, 'seed': self.random_seed}
            )
        start_state = torch.zeros(self.state_dims)

        for _ in range(self.n_generations):
            # Get initial population (parameters for each model)
            pop_params = solver.ask()
            
            for model, params in zip(models, pop_params):
                self._set_params(model, params)
            
            args = [(model, problem_instance, start_state) for model in models]
            pop_fitness = pool.starmap(self._calc_fitness, args)   
            
            # Negate fitness due to CMA-ES minimizing the cost
            solver.tell(pop_params, [-i for i in pop_fitness])
            
            max_pop_fitness = max(pop_fitness)    
            fitnesses.append(max_pop_fitness)
            avg_fitness = np.mean(fitnesses[-self.sma_window:])
            wandb.log({'Average Reward': avg_fitness})
            
            if max_pop_fitness > best_fitness:
                best_fitness = max_pop_fitness
                best_params = pop_params[pop_fitness.index(max_pop_fitness)]
                
        return best_params
    
    def _get_adaptation(self, problem_instance, best_params):
        self.attention_neuron.reset()
        self._set_params(self.attention_neuron, best_params)
        
        done = False
        num_action = 0
        action_seq = []
        state = torch.zeros(self.state_dims)
        while not done:
            num_action += 1
            action = self._select_action(state)
            reward, next_state, done = self._step(problem_instance, state, action, num_action)
            state = next_state
            action_seq += [action]
            
        return action_seq, reward
            
    # Generate optimal adaptation for a given problem instance
    def _generate_adaptations(self, problem_instance):
        self._init_wandb(problem_instance)
        
        self.attention_neuron = PINN(self.action_dims)
        
        best_params = self._fit_model(problem_instance)
        adaptation, reward = self._get_adaptation(problem_instance, best_params)
        
        wandb.log({'Adaptation': adaptation})
        wandb.log({'Reward': reward})
        wandb.finish()
        
        return adaptation