import copy
import torch
import wandb
import problems
import numpy as np

from agents.utils.ea import EA
from agents.utils.replay_buffer import MultiInputPER
from agents.utils.network import MultiInputBayesianDQN


class RL(EA):
    def __init__(self, env, num_obstacles):
        super().__init__(env, num_obstacles)
                        
        self.bdqn = None
        self.buffer = None
        self.gamma = 0.9875
        self.action_dims = self.env.observation_space.n ** 2
        
    def _init_wandb(self, problem_instance, affinity_instance):
        config = super()._init_wandb(problem_instance, affinity_instance)
        config.tau = self.tau
        config.alpha = self.alpha
        config.gamma = self.gamma 
        config.batch_size = self.batch_size
        config.action_cost = self.action_cost
        config.memory_size = self.memory_size
        config.num_episodes = self.num_episodes
        config.kl_coefficient = self.kl_coefficient
    
    def _select_action(self, onehot_state, onehot_action_sequence):    
        with torch.no_grad():
            q_values_sample = self.bdqn(torch.FloatTensor(onehot_state), torch.FloatTensor(onehot_action_sequence))
            action = torch.argmax(q_values_sample).item()
        return action
            
    def _learn(self):    
        batch, weights, tree_idxs = self.buffer.sample(self.batch_size)
        state, action_sequence, action, reward, next_state, next_action_sequence, done = batch
        
        q_values = self.bdqn(state, action_sequence)
        q = q_values.gather(1, action.long()).view(-1)

        # Select actions using online network
        next_q_values = self.bdqn(next_state, next_action_sequence)
        next_actions = next_q_values.max(1)[1].detach()
        # Evaluate actions using target network to prevent overestimation bias
        target_next_q_values = self.target_dqn(next_state, next_action_sequence)
        next_q = target_next_q_values.gather(1, next_actions.unsqueeze(1)).view(-1).detach()
        
        q_hat = reward + (1 - done) * self.gamma * next_q
        
        td_error = torch.abs(q_hat - q).detach()
        
        if weights is None:
            weights = torch.ones_like(q)

        self.bdqn.optim.zero_grad()
        # Multiply by importance sampling weights to correct bias from prioritized replay
        loss = (weights * (q_hat - q) ** 2).mean()
        
        # Form of regularization to prevent posterior collapse
        kl_divergence = 0
        for layer in [self.bdqn.state_fc1, self.bdqn.state_fc2, self.bdqn.action_fc1, self.bdqn.action_fc2, self.bdqn.combined_fc]:
            w_mu = layer.w_mu
            w_rho = layer.w_rho
            w_sigma = torch.log1p(torch.exp(w_rho))
            posterior = torch.distributions.Normal(w_mu, w_sigma)
            prior = torch.distributions.Normal(0, 1)
            
            kl_divergence += torch.distributions.kl_divergence(posterior, prior).sum()
            
        loss += self.kl_coefficient * kl_divergence
        loss.backward()
        self.bdqn.optim.step()
                
        # Update target network
        for param, target_param in zip(self.bdqn.parameters(), self.target_dqn.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return loss.item(), td_error.numpy(), tree_idxs
    
    # Train the child model on a given problem instance
    def _train(self, problem_instance, affinity_instance, start_state):
        losses = []
        rewards = []
        best_actions = None
        best_reward = -np.inf
        
        for _ in range(self.num_episodes):
            done = False
            action_lst = []
            num_actions = 0
            state = start_state
            onehot_action_sequence = np.array([0] * self.action_dims)
            while not done:
                num_actions += 1
                onehot_state = self.encoder.fit_transform(state.reshape(-1, 1)).toarray().flatten()
                action = self._select_action(onehot_state, onehot_action_sequence)
                reward, next_state, done = self._step(problem_instance, affinity_instance, state, action, num_actions)
                onehot_next_state = self.encoder.fit_transform(next_state.reshape(-1, 1)).toarray().flatten()
                onehot_next_action_sequence = onehot_action_sequence.copy()
                onehot_next_action_sequence[(num_actions-1)*64 + action] = 1
                self.buffer.add((onehot_state, onehot_action_sequence, action, reward, onehot_next_state, onehot_next_action_sequence, done))
                loss, td_error, tree_idxs = self._learn()
                state = next_state
                onehot_action_sequence = onehot_next_action_sequence
                action_lst.append(action)
            
            self.buffer.update_priorities(tree_idxs, td_error)
            
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
    def _generate_adaptations(self, problem_instance, affinity_instance):
        self._init_wandb(problem_instance, affinity_instance)
        
        self.buffer = MultiInputPER(self.state_dims, self.action_dims, 1, self.memory_size)
        self.bdqn = MultiInputBayesianDQN(self.state_dims, self.action_dims, self.num_actions, self.alpha)
        self.target_dqn = copy.deepcopy(self.bdqn)
        
        start_state = self._convert_state(problems.desc)
        best_actions, best_reward, losses, rewards = self._train(problem_instance, affinity_instance, start_state)
        
        wandb.log({'Final Reward': best_reward})
        wandb.log({'Final Actions': best_actions})
        wandb.finish()
        
        return best_actions, losses, rewards