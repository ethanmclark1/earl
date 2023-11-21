import copy
import torch
import wandb
import numpy as np
import torch.nn.functional as F

from agents.utils.ea import EA
from agents.utils.networks import Actor, Critic
from agents.utils.replay_buffer import PrioritizedReplayBuffer

from sklearn.preprocessing import OneHotEncoder


# Twin-Delayed DDPG (TD3) with Prioritized Experience Replay
class ContinuousRL(EA):
    def __init__(self, env, grid_size, num_obstacles):
        super(ContinuousRL, self).__init__(env, grid_size, num_obstacles)
                
        self.actor = None
        self.critic = None
        self.actor_target = None
        self.critic_target = None
        self.buffer = None
        
        self.gamma = 0.9875
        self.policy_freq = 2
        self.batch_size = 256
        self.noise_clamp = 0.15
        self.policy_noise = 0.05
        self.memory_size = 500000
        self.exploration_noise_start = 0.1
        self.exploration_noise_decay = 0.9999
        
        self.action_enc = OneHotEncoder(categories=[range(self.action_dims)])
        
    def _init_wandb(self, problem_instance):
        config = super()._init_wandb(problem_instance)
        config.alpha = self.alpha
        config.gamma = self.gamma
        config.batch_size = self.batch_size
        config.noise_clamp = self.noise_clamp
        config.memory_size = self.memory_size
        config.policy_freq = self.policy_freq
        config.num_episodes = self.num_episodes
        config.policy_noise = self.policy_noise
        config.exploration_noise_start = self.exploration_noise_start
        config.exploration_noise_decay = self.exploration_noise_decay
        
    def _decrement_exploration(self):
        self.exploration_noise *= self.exploration_noise_decay
        self.exploration_noise = max(self.exploration_noise, 0.01)
        
    def _select_action(self, state):
        with torch.no_grad():
            action_probs = self.actor(state)
            
            noise = torch.normal(0.0, self.exploration_noise, size=(self.action_dims,))
            noise = noise.clamp(-self.noise_clamp, self.noise_clamp)
            noisy_probs = (action_probs + noise).clamp(-1, 1)
            logits = F.softmax(noisy_probs, dim=-1)
            action = torch.multinomial(logits, 1).item()
                    
        return action
        
    def _learn(self):
        loss = {}
        loss['actor'] = None
        loss['critic'] = None

        self.total_it += 1
        
        batch, weights, tree_idxs = self.buffer.sample(self.batch_size)
        state, action, reward, next_state, done = batch
        done = done.unsqueeze(-1)
        reward = reward.unsqueeze(-1)
        
        with torch.no_grad():
            next_action_probs = self.actor_target(next_state)
            noise = torch.normal(0.0, self.exploration_noise, size=(self.batch_size, self.action_dims))
            noise = noise.clamp(-self.noise_clamp, self.noise_clamp)
            noisy_next_action_probs = (next_action_probs + noise).clamp(-1, 1)
            logits = F.softmax(noisy_next_action_probs, dim=-1)
            next_action = torch.multinomial(logits, 1)
            
            onehot_next_action = self.action_enc.fit_transform(next_action).toarray()
            next_action = torch.tensor(onehot_next_action, dtype=torch.float32)       
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q
        
        onehot_action = self.action_enc.fit_transform(action).toarray()
        onehot_action = torch.tensor(onehot_action, dtype=torch.float32)
        current_Q1, current_Q2 = self.critic(state, onehot_action)
        current_Q = torch.min(current_Q1, current_Q2)
        td_error = torch.abs(target_Q - current_Q).detach()
        
        weighted_loss_Q1 = (F.mse_loss(current_Q1, target_Q, reduction='none') * weights).mean()
        weighted_loss_Q2 = (F.mse_loss(current_Q2, target_Q, reduction='none') * weights).mean()
        loss['critic'] = weighted_loss_Q1 + weighted_loss_Q2
        
        self.critic.optim.zero_grad()
        loss['critic'].backward()
        self.critic.optim.step()
        loss['critic'] = loss['critic'].item()
        
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            loss['actor'] = -self.critic.get_Q1(state, self.actor(state)).mean()
            
            # Optimize the actor
            self.actor.optim.zero_grad()
            loss['actor'].backward()
            self.actor.optim.step()
            
            loss['actor'] = loss['actor'].item()
            
            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
        return loss, td_error.numpy(), tree_idxs
    
    def _train(self, problem_instance, start_state):
        rewards = []
        actor_losses = []
        critic_losses = []
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
                
                self.buffer.add(state, action, reward, next_state, done)
                state = next_state
                action_seq.append(action)
            
            loss, td_error, tree_idxs = self._learn()
            self.buffer.update_priorities(tree_idxs, td_error)
            
            actor_losses.append(loss['actor'])
            critic_losses.append(loss['critic'])
            rewards.append(reward)
            avg_actor_loss = np.mean(actor_losses[self.sma_window:])
            avg_critic_loss = np.mean(critic_losses[self.sma_window:])
            avg_rewards = np.mean(rewards[self.sma_window:])
            wandb.log({"Average Actor Loss": avg_actor_loss})
            wandb.log({"Average Critic Loss": avg_critic_loss})
            wandb.log({"Average Reward": avg_rewards})
            
            if reward > best_reward:
                best_actions = action_seq
                best_reward = reward
                
        return best_actions, best_reward, actor_losses, critic_losses, rewards
    
    def _generate_adaptations(self, problem_instance):
        # self._init_wandb(problem_instance)
        
        self.total_it = 0
        self.exploration_noise = self.exploration_noise_start
        
        self.actor = Actor(self.state_dims, self.action_dims, self.alpha)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = Critic(self.state_dims, self.action_dims, self.alpha)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.buffer = PrioritizedReplayBuffer(self.state_dims, self.memory_size)
        
        start_state = torch.zeros(self.state_dims)
        self._populate_buffer(problem_instance, start_state)
        best_actions, best_reward, losses, rewards = self._train(problem_instance, start_state)
        
        wandb.log({'Final Reward': best_reward})
        wandb.log({'Final Actions': best_actions})
        wandb.finish()
        
        return best_actions, losses, rewards