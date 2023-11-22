import copy
import torch
import wandb
import numpy as np

from agents.utils.ea import EA
from agents.utils.networks import BayesianDQN
from agents.utils.replay_buffer import PrioritizedReplayBuffer

# Bayesian DQN with Prioritized Experience Replay
class BDQN(EA):
    def __init__(self, env, grid_size, num_obstacles):
        super(BDQN, self).__init__(env, grid_size, num_obstacles)
                        
        self.bdqn = None
        self.buffer = None
        
        self.tau = 0.008
        self.alpha = 0.002
        self.gamma = 0.9875
        self.batch_size = 256
        self.num_episodes = 500
        self.dummy_episodes = 25
        self.kl_coefficient = 0.001
        self.memory_size = 10000000
        
    def _init_wandb(self, problem_instance):
        config = super()._init_wandb(problem_instance)
        config.tau = self.tau
        config.alpha = self.alpha
        config.gamma = self.gamma 
        config.batch_size = self.batch_size
        config.memory_size = self.memory_size
        config.num_episodes = self.num_episodes
        config.dummy_episodes = self.dummy_episodes
        config.kl_coefficient = self.kl_coefficient
    
    def _select_action(self, state):    
        with torch.no_grad():
            q_values_sample = self.bdqn(state)
            action = torch.argmax(q_values_sample).item()
        return action
            
    def _learn(self):    
        batch, weights, tree_idxs = self.buffer.sample(self.batch_size)
        state, action, reward, next_state, done = batch
        
        q_values = self.bdqn(state)
        q = q_values.gather(1, action.long()).view(-1)

        # Select actions using online network
        next_q_values = self.bdqn(next_state)
        next_actions = next_q_values.max(1)[1].detach()
        # Evaluate actions using target network to prevent overestimation bias
        target_next_q_values = self.target_dqn(next_state)
        next_q = target_next_q_values.gather(1, next_actions.unsqueeze(1)).view(-1).detach()
        
        q_hat = reward + (1 - done) * self.gamma * next_q
        
        td_error = torch.abs(q_hat - q).detach()
        
        if weights is None:
            weights = torch.ones_like(q)

        self.bdqn.optim.zero_grad()
        # Multiply by importance sampling weights to correct bias from prioritized replay
        loss = (weights * (q_hat - q) ** 2).mean()
        
        # Regularize to prevent posterior collapse
        kl_divergence = 0
        for layer in self.bdqn.children():
            # Penalize the network for deviating from the prior
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
    
    def _train(self, problem_instance, start_state):
        losses = []
        rewards = []
        best_actions = None
        best_reward = -np.inf
        
        for _ in range(self.num_episodes):
            done = False
            num_action = 0
            action_seq = []
            state = start_state
            while not done:
                num_action += 1
                action = self._select_action(state)
                reward, next_state, done = self._step(problem_instance, state, action, num_action)    
                
                state = next_state
                action_seq += [action]
                
            self.buffer.hallucinate(action_seq, reward)
            loss, td_error, tree_idxs = self._learn()
            self.buffer.update_priorities(tree_idxs, td_error)
            
            losses.append(loss)
            rewards.append(reward)
            avg_loss = np.mean(losses[self.sma_window:])
            avg_rewards = np.mean(rewards[self.sma_window:])
            wandb.log({"Average Loss": avg_loss})
            wandb.log({"Average Reward": avg_rewards})
            
            if reward > best_reward:
                best_actions = action_seq
                best_reward = reward
                
        return best_actions, best_reward
    
    # Generate optimal adaptation for a given problem instance
    def _generate_adaptations(self, problem_instance):
        self._init_wandb(problem_instance)
        
        self.bdqn = BayesianDQN(self.state_dims, self.action_dims, self.alpha)
        self.target_dqn = copy.deepcopy(self.bdqn)
        self.buffer = PrioritizedReplayBuffer(self.state_dims, self.memory_size)
        
        start_state = torch.zeros(self.state_dims)
        self._populate_buffer(problem_instance, start_state)
        best_actions, best_reward = self._train(problem_instance, start_state)
        
        wandb.log({'Final Reward': best_reward})
        wandb.log({'Final Actions': best_actions})
        wandb.finish()
        
        return best_actions