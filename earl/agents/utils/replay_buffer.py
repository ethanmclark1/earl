# This code is based on the following repository:
# https://github.com/Howuhh/prioritized_experience_replay
# Author: Alexander Nikulin (Howuhh)
# Title: Prioritized Experience Replay - Memory: buffer.py
# Version: 339e6aa

import torch
import random
import itertools
import numpy as np

from agents.utils.sumtree import SumTree


class PrioritizedReplayBuffer:
    def __init__(self, state_size, buffer_size):
        self.tree = SumTree(size=buffer_size)

        # Degree of prioritization
        self.alpha = 0.1
        # Degree of bias correction for importance sampling weights
        self.beta = 0.1 
        # Annealing factor for beta
        self.beta_increment = 1e-3
        # Small value to ensure that no transition has zero priority
        self.epsilon = 1e-2
        # Keep track of maximum priority to assign to new experiences
        self.max_priority = 1e-2
        # Keep track of the indices of combinations in the buffer
        self.indices = {}
        
        self.state_dims = state_size
        
        self.state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.action = torch.empty(buffer_size, 1, dtype=torch.float)
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.done = torch.empty(buffer_size, dtype=torch.int)

        # Manging buffer size and current position
        self.count = 0
        self.real_size = 0
        self.size = buffer_size
    
    def _sample_permutations(self, action_seq):
        max_seq_len = 8
        permutations = set()
        permutations.add(tuple(action_seq))
        tmp_action_seq = action_seq.copy()
        terminating_action = tmp_action_seq.pop()
        
        # If the action sequence is short, we can afford to sample all permutations
        if len(tmp_action_seq) <= max_seq_len:
            permutations_no_termination = itertools.permutations(tmp_action_seq)
            permutations = {permutation + (terminating_action,) for permutation in permutations_no_termination}
        else:
            while len(permutations) < np.math.factorial(max_seq_len):
                random.shuffle(tmp_action_seq)
                permutations.add(tuple(tmp_action_seq + [terminating_action]))
                
        return list(permutations)

    # Combination is only used for CMDP
    def add(self, state, action, reward, next_state, done):
        # Add with maximum priority to ensure that every transition is sampled at least once        
        self.tree.add(self.max_priority, self.count)
        
        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)
            
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)
        
    # Hallucinate episodes by permuting the action sequence to simulate commutativity
    def hallucinate(self, action_seq, reward):        
        start_state = np.array([0] * self.state_dims)
        permutations = self._sample_permutations(action_seq)
        for permutation in permutations:
            state = start_state
            for action in permutation:
                next_state = state.copy()
                # Terminating action does not change state
                if action != len(state):
                    next_state[action] = 4
                
                self.add(state, action, 0, next_state, 0)
            
            self.reward[self.count-1] = torch.as_tensor(reward)
            self.done[self.count-1] = torch.as_tensor(1)

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