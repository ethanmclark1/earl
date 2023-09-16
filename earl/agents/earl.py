import os
import copy
import torch
import wandb
import pickle
import problems
import numpy as np
import networkx as nx

from itertools import chain
from agents.utils.network import DuelingDQN
from agents.utils.replay_buffer import PrioritizedReplayBuffer

class EARL:
    def __init__(self, env, num_obstacles):
        self._init_hyperparams()
                
        self.env = env
        self.dqn = None
        self.buffer = None
        self.state_dims = 64
        self.num_actions = 64
        self.configs_to_consider = 250
        self.num_cols = env.unwrapped.ncol
        self.num_obstacles = num_obstacles
        
        self.rng = np.random.default_rng(seed=42)
        
        self.max_actions = 8
        self.action_set = set()
        self.mapping = {'F': 0, 'H': 1, 'S': 2, 'G': 3, 'O': 4}
        
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
        num_records = 10
        
        self.tau = 5e-3
        self.alpha = 1e-4
        self.gamma = 0.99
        self.batch_size = 256
        self.granularity = 0.20
        self.memory_size = 30000
        self.epsilon_start = 1.0
        self.dummy_episodes = 200
        self.num_episodes = 15000
        self.epsilon_decay = 0.9997
        self.record_freq = self.num_episodes // num_records
        
    def _init_wandb(self, problem_instance):
        wandb.init(project='earl', entity='ethanmclark1', name=problem_instance)
        config = wandb.config
        config.tau = self.tau
        config.alpha = self.alpha
        config.gamma = self.gamma 
        config.epsilon = self.epsilon_start
        config.batch_size = self.batch_size
        config.granularity = self.granularity
        config.memory_size = self.memory_size
        config.num_episodes = self.num_episodes
        config.epsilon_decay = self.epsilon_decay
        config.dummy_episodes = self.dummy_episodes
        
    def _decrement_exploration(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(0.01, self.epsilon)
    
    # Convert map description to encoded array
    def _convert_state(self, state):
        flattened_state = np.array(list(chain.from_iterable(state)))
        numerical_state = np.vectorize(self.mapping.get)(flattened_state)
        return numerical_state
                
    def _select_action(self, state):
        with torch.no_grad():
            if self.rng.random() > self.epsilon:
                action = self.rng.integers(self.num_actions)
            else:
                q_vals = self.dqn(torch.FloatTensor(state))
                action = torch.argmax(q_vals).item()
        return action
    
    # Populate buffer with dummy transitions
    def _populate_buffer(self, problem_instance, start_state):
        for _ in range(self.dummy_episodes):
            done = False
            num_action = 0
            state = start_state
            while not done:
                num_action += 1
                action = self._select_action(state)
                reward, next_state, done, _ = self._step(problem_instance, state, action, num_action)
                self.buffer.add((state, action, reward, next_state, done))
                state = next_state

    # TODO: Implement reward function
    def _get_reward(self, problem_instance, state):
        desc = state.reshape((8, 8))
        for _ in range(self.configs_to_consider):
            start, goal, obstacles = problems.get_entity_positions(problem_instance, self.num_obstacles)
            desc[start] = self.mapping['S']
            desc[goal] = self.mapping['G']
            for obstacle in obstacles:
                desc[obstacle] = self.mapping['H']
            a=3
    
    # Apply reconfiguration to task environment
    def _step(self, problem_instance, state, action, num_actions):
        reward = 0
        done = False   
        prev_num_actions = len(self.action_set)
        
        next_state = state
        next_state[action] = self.mapping['O']
        self.action_set.add(action)
        
        if len(self.action_set) != prev_num_actions or num_actions == self.max_actions:
            done = True
            reward = self._get_reward(problem_instance, next_state)  
            self.action_set.clear()  
        
        return reward, next_state, done
            
    def _learn(self):    
        batch, weights, tree_idxs = self.buffer.sample(self.batch_size)
        state, action, reward, next_state, done = batch
        
        q_values = self.dqn(state)
        q = q_values.gather(1, action.long()).view(-1)

        # Select actions using online network
        next_q_values = self.dqn(next_state)
        next_actions = next_q_values.max(1)[1].detach()
        # Evaluate actions using target network to prevent overestimation bias
        target_next_q_values = self.target_dqn(next_state)
        next_q = target_next_q_values.gather(1, next_actions.unsqueeze(1)).view(-1).detach()
        
        q_hat = reward + (1 - done) * self.gamma * next_q
        
        td_error = torch.abs(q_hat - q).detach()
        
        if weights is None:
            weights = torch.ones_like(q)

        self.dqn.optim.zero_grad()
        # Multiply by importance sampling weights to correct bias from prioritized replay
        loss = (weights * (q_hat - q) ** 2).mean()
        loss.backward()
        self.dqn.optim.step()
                
        # Update target network
        for param, target_param in zip(self.dqn.parameters(), self.target_dqn.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return loss.item(), td_error.numpy(), tree_idxs
    
    # Train the child model on a given problem instance
    def _train(self, problem_instance, start_state):
        losses = []
        returns = []
        best_reward = -np.inf
        best_adaptive_actions = None
        
        for _ in range(self.num_episodes):
            done = False
            action_lst = []
            state = start_state
            num_actions = 0
            while not done:
                num_actions += 1
                action = self._select_action(state)
                reward, next_state, done, _ = self._step(problem_instance, action, num_actions)
                self.buffer.add((state, action, reward, next_state, done))
                loss, td_error, tree_idxs = self._learn()
                state = next_state
                action_lst.append(action)
            
            self.buffer.update_priorities(tree_idxs, td_error)
            self._decrement_exploration()
            
            losses.append(loss)
            returns.append(reward)
            avg_loss = np.mean(losses[-100:])
            avg_returns = np.mean(returns[-100:])
            wandb.log({"Average Value Loss": avg_loss})
            wandb.log({"Average Returns": avg_returns})
            
            if reward > best_reward:
                best_reward = reward
                best_adaptive_actions = action_lst
                
        return best_adaptive_actions, best_reward
    
    # Generate optimal reconfiguration for a given problem instance
    def _generate_reconfiguration(self, problem_instance):
        self._init_wandb(problem_instance)
        
        self.epsilon = self.epsilon_start
        self.buffer = PrioritizedReplayBuffer(self.state_dims, 1, self.memory_size)
        self.dqn = DuelingDQN(self.state_dims, self.num_actions, self.alpha)
        self.target_dqn = copy.deepcopy(self.dqn)
        
        start_state = self._convert_state(problems.desc)
        self._populate_buffer(problem_instance, start_state)
        best_adaptive_actions, best_reward = self._train(problem_instance, start_state)
        
        wandb.log({'Final Reward': best_reward})
        wandb.log({'Final Adaptive Actions': best_adaptive_actions})
        wandb.finish()
        
        return best_adaptive_actions
        
    def get_reconfigured_env(self, problem_instance):
        try:
            reconfiguration = self._load(problem_instance)
        except FileNotFoundError:
            print(f'No stored reconfiguration for {problem_instance} problem instance.')
            print('Generating new reconfiguration...\n')
            reconfiguration = self._generate_reconfiguration(problem_instance)
            self._save(problem_instance, reconfiguration)
        
        return reconfiguration