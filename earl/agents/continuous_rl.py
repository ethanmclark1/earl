import copy
import torch
import wandb
import numpy as np
import torch.nn.functional as F

from agents.utils.ea import EA
from agents.utils.networks import Actor, Critic
from agents.utils.replay_buffer import PrioritizedReplayBuffer


class ContinuousRL(EA):
    def __init__(self, env, grid_size, num_obstacles):
        super(ContinuousRL, self).__init__(env, grid_size, num_obstacles)
        self.action_dim = 3
        
        self.actor = None
        self.actor_target = None
        self.critic = None
        self.critic_target = None
        
    def _init_wandb(self, problem_instance):
        config = super()._init_wandb(problem_instance)
        
    def _decrement_exploration(self):
        self.exploration_noise *= self.exploration_noise_decay
        self.exploration_noise = max(self.exploration_noise, 0.01)
        
    def _select_action(self, state):
        with torch.no_grad():
            action = self.actor(state)
            
            noise = self.rng.normal(0, self.exploration_noise, size=self.action_dim)
            action = (action.detach().numpy() + noise).clip(-1, 1)
        
        return action
        
    def _learn(self):
        actor_loss = None
        self.total_it += 1
        
        batch, weights, tree_idxs = self.buffer.sample(self.batch_size)
        state, action, reward, next_state, done = batch
        
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)
            
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q
        
        current_Q1, current_Q2 = self.critic(state, action)
        current_Q = torch.min(current_Q1, current_Q2)
        td_error = torch.abs(target_Q - current_Q).detach()
        
        weighted_loss_Q1 = (F.mse_loss(current_Q1, target_Q, reduction='none') * weights).mean()
        weighted_loss_Q2 = (F.mse_loss(current_Q2, target_Q, reduction='none') * weights).mean()
        critic_loss = weighted_loss_Q1 + weighted_loss_Q2
        
        self.critic.optim.zero_grad()
        critic_loss.backward()
        self.critic.optim.step()
        
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.get_Q1(state, self.actor(state)).mean()
            
            # Optimize the actor
            self.actor.optim.zero_grad()
            actor_loss.backward()
            self.actor.optim.step()
            
            actor_loss = actor_loss.item()
            
            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
        return actor_loss, critic_loss.item(), td_error.numpy(), tree_idxs
    
    def _train(self, problem_instance, start_state):
        losses = []
        rewards = []
        best_actions = None
        best_reward = -np.inf
        
        for _ in range(self.num_episodes):
            done = False
            action_seq = []
            num_action = 0
            state = start_state
            while not done:
                num_action += 1
                action = self._select_action(state)
                reward, next_state, done = self._step(problem_instance, state, action, num_action)
                
                self.buffer.add(state, action, reward, next_state, done)
                state = next_state
                action_seq.append(action)
            
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
                
        return best_actions, best_reward, losses, rewards
    
    def _generate_adaptations(self, problem_instance):
        # self._init_wandb(problem_instance)
        
        self.total_it = 0
        self.exploration_noise = self.exploration_noise_start
        
        self.actor = Actor(self.state_dim, self.action_dim)
        self.actor_target = Actor(self.state_dim, self.action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(self.state_dim, self.action_dim)
        self.critic_target = Critic(self.state_dim, self.action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.buffer = PrioritizedReplayBuffer(self.memory_size, self.alpha)
        
        start_state = torch.zeros(self.state_dim)
        best_actions, best_reward, losses, rewards = self._train(problem_instance, start_state)
        
        wandb.log({'Final Reward': best_reward})
        wandb.log({'Final Actions': best_actions})
        wandb.finish()
        
        return best_actions, losses, rewards
        
    
        
    