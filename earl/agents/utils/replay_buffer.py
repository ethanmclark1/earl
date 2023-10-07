# This code is based on the following repository:
# https://github.com/Howuhh/prioritized_experience_replay
# Author: Alexander Nikulin (Howuhh)
# Title: Prioritized Experience Replay - Memory: buffer.py
# Version: 339e6aa

import torch
import random

from agents.utils.sumtree import SumTree


class PrioritizedReplayBuffer:
    def __init__(self, state_size, action_size, buffer_size, epsilon=1e-2, alpha=0.1, beta=0.1, beta_increment=1e-3):
        self.tree = SumTree(size=buffer_size)

        # Degree of prioritization
        self.alpha = alpha
        # Degree of bias correction for importance sampling weights
        self.beta = beta 
        # Annealing factor for beta
        self.beta_increment = beta_increment
        # Small value to ensure that no transition has zero priority
        self.epsilon = epsilon  
        # Keep track of maximum priority to assign to new experiences
        self.max_priority = epsilon

        self.state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.action = torch.empty(buffer_size, action_size, dtype=torch.float)
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.done = torch.empty(buffer_size, dtype=torch.int)

        # Manging buffer size and current position
        self.count = 0
        self.real_size = 0
        self.size = buffer_size

    def add(self, transition):
        # Add with maximum priority to ensure that every transition is sampled at least once
        self.tree.add(self.max_priority, self.count)
        
        state, action, reward, next_state, done = transition

        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    # Sample batch of transitions with importance sampling weights
    def sample(self, batch_size):
        self.beta = min(1., self.beta + self.beta_increment)
        
        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)

        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a, b)

            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        probs = priorities / self.tree.total

        weights = (self.real_size * probs) ** -self.beta

    
        weights = weights / weights.max()

        batch = (
            self.state[sample_idxs],
            self.action[sample_idxs],
            self.reward[sample_idxs],
            self.next_state[sample_idxs],
            self.done[sample_idxs]
        )
        
        return batch, weights, tree_idxs

    # Update priorities of sampled transitions
    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for data_idx, priority in zip(data_idxs, priorities):
            priority = (priority + self.epsilon) ** self.alpha

            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)
            
            
class MultiInputPER(PrioritizedReplayBuffer):
    def __init__(self, state_size, action_sequence_size, action_size, buffer_size, epsilon=1e-2, alpha=0.1, beta=0.1, beta_increment=1e-3): 
        super(MultiInputPER, self).__init__(state_size, action_size, buffer_size, epsilon, alpha, beta, beta_increment)
        
        self.action_sequence = torch.empty(buffer_size, action_sequence_size, dtype=torch.float)
        self.next_action_sequence = torch.empty(buffer_size, action_sequence_size, dtype=torch.float)
        
    def add(self, transition):
        # Add with maximum priority to ensure that every transition is sampled at least once
        self.tree.add(self.max_priority, self.count)
        
        state, action_sequence, action, reward, next_state, next_action_sequence, done = transition

        self.state[self.count] = torch.as_tensor(state)
        self.action_sequence[self.count] = torch.as_tensor(action_sequence)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.next_action_sequence[self.count] = torch.as_tensor(next_action_sequence)
        self.done[self.count] = torch.as_tensor(done)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    # Sample batch of transitions with importance sampling weights
    def sample(self, batch_size):
        self.beta = min(1., self.beta + self.beta_increment)
        
        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)

        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a, b)

            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        probs = priorities / self.tree.total

        weights = (self.real_size * probs) ** -self.beta

    
        weights = weights / weights.max()

        batch = (
            self.state[sample_idxs],
            self.action_sequence[sample_idxs],
            self.action[sample_idxs],
            self.reward[sample_idxs],
            self.next_state[sample_idxs],
            self.next_action_sequence[sample_idxs],
            self.done[sample_idxs]
        )
        
        return batch, weights, tree_idxs