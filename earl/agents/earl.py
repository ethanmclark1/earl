import os
import copy
import torch
import wandb
import pickle
import problems
import numpy as np
import networkx as nx

from itertools import chain
from sklearn.preprocessing import OneHotEncoder

from agents.utils.network import DuelingDQN
from agents.utils.replay_buffer import PrioritizedReplayBuffer

class EARL:
    def __init__(self, env, num_obstacles):
        self._init_hyperparams()
        
        self.max_actions = 5
        self.action_set = set()
        self.mapping = {'F': 0, 'H': 1, 'S': 2, 'G': 3, 'T': 4}
                
        self.env = env
        self.dqn = None
        self.buffer = None
        self.configs_to_consider = 500
        self.num_cols = env.unwrapped.ncol
        self.num_obstacles = num_obstacles
        self.num_actions = env.observation_space.n
        self.state_dims = env.observation_space.n * len(self.mapping)
        
        self.rng = np.random.default_rng(seed=42)
        self.encoder = OneHotEncoder(categories=[range(len(self.mapping))])
        
    def _save(self, problem_instance, reconfiguration):
        directory = 'earl/agents/history'
        filename = f'{problem_instance}.pkl'
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
        self.memory_size = 10000
        self.epsilon_start = 1.0
        self.num_episodes = 5000
        self.dummy_episodes = 200
        self.epsilon_decay = 0.9997
        self.record_freq = self.num_episodes // num_records
        
    def _init_wandb(self, problem_instance):
        wandb.init(project='earl', entity='ethanmclark1', name=problem_instance.capitalize())
        config = wandb.config
        config.tau = self.tau
        config.alpha = self.alpha
        config.gamma = self.gamma 
        config.epsilon = self.epsilon_start
        config.batch_size = self.batch_size
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
    
    def _select_action(self, onehot_state):
        with torch.no_grad():
            if self.rng.random() < self.epsilon:
                action = self.rng.integers(self.num_actions)
            else:
                q_vals = self.dqn(torch.FloatTensor(onehot_state))
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
                onehot_state = self.encoder.fit_transform(state.reshape(-1, 1)).toarray().flatten()
                action = self._select_action(onehot_state)
                reward, next_state, done = self._step(problem_instance, state, action, num_action)
                onehot_next_state = self.encoder.fit_transform(next_state.reshape(-1, 1)).toarray().flatten()
                self.buffer.add((onehot_state, action, reward, onehot_next_state, done))
                state = next_state

    # Calculate rewards for a given state by averaging A* path lengths over multiple trials
    def _get_reward(self, problem_instance, state):
        def manhattan_dist(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        rewards = []
        desc = copy.deepcopy(state).reshape((8, 8))
        row_idx, col_idx = np.where(desc == self.mapping['T'])
        transporters = set(zip(row_idx, col_idx))
        
        for _ in range(self.configs_to_consider):
            graph = nx.grid_graph(dim=[self.num_cols, self.num_cols])
            start, goal, obstacles = problems.get_entity_positions(problem_instance, self.num_obstacles)
            desc[start] = self.mapping['S']
            desc[goal] = self.mapping['G']
            for obstacle in obstacles:
                if desc[obstacle] != self.mapping['T']:
                    desc[obstacle] = self.mapping['H']
            
            for i in range(self.num_cols):
                for j in range(self.num_cols):
                    cell_value = desc[i, j]
                    for neighbor in graph.neighbors((i, j)):
                        weight = 6
                        if cell_value == self.mapping['H']:
                            weight = 100
                        elif cell_value == self.mapping['T']:
                            weight = 0
                        graph[(i, j)][neighbor]['weight'] = weight
                        
            path = set(nx.astar_path(graph, start, goal, manhattan_dist, 'weight'))
            rewards += [-len(path - transporters)]

        avg_reward = np.mean(rewards)
        return avg_reward
    
    # Apply reconfiguration to task environment
    def _step(self, problem_instance, state, action, num_actions):
        reward = 0
        done = False   
        state = copy.deepcopy(state)
        prev_num_actions = len(self.action_set)
        
        next_state = state
        next_state[action] = self.mapping['T']
        self.action_set.add(action)
        
        if len(self.action_set) == prev_num_actions or num_actions == self.max_actions:
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
        best_actions = None
        best_reward = -np.inf
        
        for _ in range(self.num_episodes):
            done = False
            action_lst = []
            num_actions = 0
            state = start_state
            while not done:
                num_actions += 1
                onehot_state = self.encoder.fit_transform(state.reshape(-1, 1)).toarray().flatten()
                action = self._select_action(onehot_state)
                reward, next_state, done = self._step(problem_instance, state, action, num_actions)
                onehot_next_state = self.encoder.fit_transform(next_state.reshape(-1, 1)).toarray().flatten()
                self.buffer.add((onehot_state, action, reward, onehot_next_state, done))
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
                best_actions = action_lst
                best_reward = reward
                
        return best_actions, best_reward
    
    # Generate optimal reconfiguration for a given problem instance
    def _generate_reconfigurations(self, problem_instance):
        self._init_wandb(problem_instance)
        
        self.epsilon = self.epsilon_start
        self.buffer = PrioritizedReplayBuffer(self.state_dims, 1, self.memory_size)
        self.dqn = DuelingDQN(self.state_dims, self.num_actions, self.alpha)
        self.target_dqn = copy.deepcopy(self.dqn)
        
        start_state = self._convert_state(problems.desc)
        self._populate_buffer(problem_instance, start_state)
        best_actions, best_reward = self._train(problem_instance, start_state)
        
        wandb.log({'Final Reward': best_reward})
        wandb.log({'Final Actions': best_actions})
        wandb.finish()
        
        return best_actions
    
    # Get reconfigurations for a given problem instance
    def get_reconfigurations(self, problem_instance):
        try:
            reconfigurations = self._load(problem_instance)
        except FileNotFoundError:
            print(f'No stored reconfiguration for {problem_instance.capitalize()} problem instance.')
            print('Generating new reconfiguration...\n')
            reconfigurations = self._generate_reconfigurations(problem_instance)
            self._save(problem_instance, reconfigurations)
        
        return reconfigurations
    
    # Apply reconfigurations to task environment
    def get_reconfigured_env(self, desc, reconfigurations):
        for reconfigruation in reconfigurations:
            row = reconfigruation // self.num_cols
            col = reconfigruation % self.num_cols
            if desc[row, col] == b'S' or desc[row, col] == b'G':
                continue
            desc[row, col] = b'T'
        
        return desc