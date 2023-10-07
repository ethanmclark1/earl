import copy
import torch
import wandb
import problems
import numpy as np

from agents.utils.ea import EA
from agents.utils.network import MultiInputDuelingDQN
from agents.utils.replay_buffer import MultiInputPER


class RL(EA):
    def __init__(self, env, num_obstacles):
        super().__init__(env, num_obstacles)
        self._init_hyperparams()
                
        self.dqn = None
        self.buffer = None
        self.action_dims = self.env.observation_space.n * self.max_actions
    
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
        config = super()._init_wandb(problem_instance)
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
    
    def _select_action(self, onehot_state, onehot_action_sequence):
        with torch.no_grad():
            if self.rng.random() < self.epsilon:
                action = self.rng.integers(self.num_actions)
            else:
                q_vals = self.dqn(torch.FloatTensor(onehot_state), torch.FloatTensor(onehot_action_sequence))
                action = torch.argmax(q_vals).item()
        return action
    
    # Populate buffer with dummy transitions
    def _populate_buffer(self, problem_instance, start_state):
        for _ in range(self.dummy_episodes):
            done = False
            num_action = 0
            state = start_state
            onehot_action_sequence = np.array([0] * self.action_dims)
            while not done:
                num_action += 1
                onehot_state = self.encoder.fit_transform(state.reshape(-1, 1)).toarray().flatten()
                action = self._select_action(onehot_state, onehot_action_sequence)
                reward, next_state, done = self._step(problem_instance, state, action, num_action)
                onehot_next_state = self.encoder.fit_transform(next_state.reshape(-1, 1)).toarray().flatten()
                onehot_next_action_sequence = onehot_action_sequence.copy()
                onehot_next_action_sequence[(num_action-1)*64 + action] = 1
                self.buffer.add((onehot_state, onehot_action_sequence, action, reward, onehot_next_state, onehot_next_action_sequence, done))
                state = next_state
                onehot_action_sequence = onehot_next_action_sequence
            
    def _learn(self):    
        batch, weights, tree_idxs = self.buffer.sample(self.batch_size)
        state, action_sequence, action, reward, next_state, next_action_sequence, done = batch
        
        q_values = self.dqn(state, action_sequence)
        q = q_values.gather(1, action.long()).view(-1)

        # Select actions using online network
        next_q_values = self.dqn(next_state, next_action_sequence)
        next_actions = next_q_values.max(1)[1].detach()
        # Evaluate actions using target network to prevent overestimation bias
        target_next_q_values = self.target_dqn(next_state, next_action_sequence)
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
        rewards = []
        best_actions = None
        best_reward = -np.inf
        
        for _ in range(self.num_episodes):
            done = False
            action_lst = []
            num_action = 0
            state = start_state
            onehot_action_sequence = np.array([0] * self.action_dims)
            while not done:
                num_action += 1
                onehot_state = self.encoder.fit_transform(state.reshape(-1, 1)).toarray().flatten()
                action = self._select_action(onehot_state, onehot_action_sequence)
                reward, next_state, done = self._step(problem_instance, state, action, num_action)
                onehot_next_state = self.encoder.fit_transform(next_state.reshape(-1, 1)).toarray().flatten()
                onehot_next_action_sequence = onehot_action_sequence.copy()
                onehot_next_action_sequence[(num_action-1)*64 + action] = 1
                self.buffer.add((onehot_state, onehot_action_sequence, action, reward, onehot_next_state, onehot_next_action_sequence, done))
                loss, td_error, tree_idxs = self._learn()
                state = next_state
                onehot_action_sequence = onehot_next_action_sequence
                action_lst.append(action)
            
            self.buffer.update_priorities(tree_idxs, td_error)
            self._decrement_exploration()
            
            losses.append(loss)
            rewards.append(reward)
            avg_loss = np.mean(losses[-100:])
            avg_rewards = np.mean(rewards[-100:])
            wandb.log({"Average Value Loss": avg_loss})
            wandb.log({"Average Rewards": avg_rewards})
            
            if reward > best_reward:
                best_actions = action_lst
                best_reward = reward
                
        return best_actions, best_reward, losses, rewards
    
    # Generate optimal adaptation for a given problem instance
    def _generate_adaptations(self, problem_instance):
        # self._init_wandb(problem_instance)
        
        self.epsilon = self.epsilon_start
        self.buffer = MultiInputPER(self.state_dims, self.action_dims, 1, self.memory_size)
        self.dqn = MultiInputDuelingDQN(self.state_dims, self.action_dims, self.num_actions, self.alpha)
        self.target_dqn = copy.deepcopy(self.dqn)
        
        start_state = self._convert_state(problems.desc)
        self._populate_buffer(problem_instance, start_state)
        best_actions, best_reward, losses, rewards = self._train(problem_instance, start_state)
        
        wandb.log({'Final Reward': best_reward})
        wandb.log({'Final Actions': best_actions})
        wandb.finish()
        
        return best_actions, losses, rewards