import os
import math
import copy
import wandb
import torch
import itertools
import numpy as np
import torch.nn.functional as F

from agents.utils.ea import EA
from agents.utils.networks import RewardEstimator
from agents.utils.buffers import RewardBuffer, CommutativeRewardBuffer

torch.manual_seed(seed=42)


class BasicQTable(EA):
    def __init__(self, env, random_state, reward_prediction_type, rng):
        super(BasicQTable, self).__init__(env, random_state, rng)
        self._init_hyperparams()
        
        self.name = self.__class__.__name__
        self.output_dir = f'earl/agents/history/estimator/{self.name.lower()}'
                
        self.q_table = None
        self.reward_estimator = None
        self.target_reward_estimator = None
        self.reward_buffer = None
        self.commutative_reward_buffer = None
        self.reward_prediction_type = reward_prediction_type
        
        self.nS = 2 ** self.state_dims
        
    def _init_hyperparams(self):        
        # Reward Estimator
        self.batch_size = 128
        self.dropout_rate = 0.40
        self.estimator_tau = 0.25
        self.estimator_alpha = 0.0003
        self.model_save_interval = 25000

        # Q-Table
        self.alpha = 0.0005
        self.epsilon_start = 1
        self.min_epsilon = 0.10
        self.max_seq_len = 7 if self.problem_size == '8x8' else 5
        self.sma_window = 5000 if self.problem_size == '8x8' else 1000
        self.num_episodes = 500000 if self.problem_size == '8x8' else 100000
        self.epsilon_decay = 0.0001 if self.problem_size == '8x8' else 0.00001
        
        # Evaluation Settings
        self.eval_window = 100
        self.eval_configs = 20
        self.eval_episodes = 10
        self.eval_freq = 1000 if self.problem_size == '8x8' else 500
        
        
    def _init_wandb(self, problem_instance):
        config = super()._init_wandb(problem_instance)
        config.alpha = self.alpha
        config.eval_freq = self.eval_freq
        config.batch_size = self.batch_size
        config.sma_window = self.sma_window
        config.max_action = self.max_action
        config.batch_size = self.batch_size
        config.eval_window = self.eval_window
        config.action_cost = self.action_cost
        config.min_epsilon = self.min_epsilon
        config.max_seq_len = self.max_seq_len
        config.random_state = self.random_state
        config.num_episodes = self.num_episodes
        config.dropout_rate = self.dropout_rate
        config.problem_size = self.problem_size
        config.eval_episodes = self.eval_episodes
        config.epsilon_decay = self.epsilon_decay
        config.estimator_tau = self.estimator_tau
        config.warmup_episodes = self.warmup_episodes
        config.estimator_alpha = self.estimator_alpha
        config.reward_estimator = self.reward_estimator 
        config.action_success_rate = self.action_success_rate
        config.model_save_interval = self.model_save_interval
        config.configs_to_consider = self.configs_to_consider
        config.reward_prediction_type = self.reward_prediction_type
    
    # Only decrement epsilon after reward estimator has been sufficiently trained
    def _decrement_epsilon(self, episode):
        if 'approximate' not in self.reward_prediction_type or episode > self.warmup_episodes:
            self.epsilon -= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.min_epsilon)
            
    def _select_action(self, state):
        if self.rng.random() < self.epsilon:
            action = self.rng.integers(self.action_dims)
        else:
            state_idx = self._get_state_idx(state)
            action = self.q_table[state_idx].argmax()
            
        return action
    
    def _add_transition(self, state, action, reward, next_state, prev_state, prev_action, prev_reward):
        state_idx = self._get_state_idx(state)
        next_state_idx = self._get_state_idx(next_state)
        
        state_proxy = self._get_mutable_state(state)
        next_state_proxy = self._get_mutable_state(next_state)
        
        self.reward_buffer.add(state_proxy, action, reward, next_state_proxy)
        
        if prev_state is not None:
            prev_state_idx = self._get_state_idx(prev_state)
            prev_state_proxy = self._get_mutable_state(prev_state)
            
            action_a_success = prev_state_idx != state_idx
            action_b_success = state_idx != next_state_idx
            
            transformed_action = self._transform_action(action)
            commutative_state = self._place_bridge(prev_state, transformed_action)
            commutative_state_proxy = self._get_mutable_state(commutative_state)
            
            if action_a_success and action_b_success:            
                self.commutative_reward_buffer.add(prev_state_proxy, action, prev_reward, commutative_state_proxy, prev_action, reward, next_state_proxy)
            elif action_a_success:
                self.commutative_reward_buffer.add(prev_state_proxy, action, prev_reward, prev_state_proxy, prev_action, reward, state_proxy)
            elif action_b_success:
                self.commutative_reward_buffer.add(prev_state_proxy, action, prev_reward, commutative_state_proxy, prev_action, reward, commutative_state_proxy)
            else:
                self.commutative_reward_buffer.add(prev_state_proxy, action, prev_reward, prev_state_proxy, prev_action, reward, prev_state_proxy)
                                
    def _update_q_table(self, state, action, reward, next_state, done, losses, traditional_update=True):
        if 'approximate' in self.reward_prediction_type:
            with torch.no_grad():
                state_proxy = self._get_mutable_state(state)
                next_state_proxy = self._get_mutable_state(next_state)
                
                state_tensor = torch.as_tensor(state_proxy, dtype=torch.float)
                action_tensor = torch.as_tensor([action], dtype=torch.float)
                next_state_tensor = torch.as_tensor(next_state_proxy, dtype=torch.float)
                
                step = torch.cat([state_tensor, action_tensor, next_state_tensor], dim=-1)
                reward = self.target_reward_estimator(step).item()
        
        state = self._get_state_idx(state)
        next_state = self._get_state_idx(next_state)
        
        td_target = reward + (1 - done) * self.q_table[next_state].max() 
        td_error = td_target - self.q_table[state, action]
        
        self.q_table[state, action] += self.alpha * td_error
        
        if traditional_update:
            losses['traditional_td_error'] += abs(td_error)
        else:
            losses['commutative_td_error'] += abs(td_error)
            
        return losses
    
    def _update_estimator(self, losses, traditional_update=True):
        if self.reward_buffer.real_size < self.batch_size:
            return losses
        
        self.reward_estimator.optim.zero_grad(set_to_none=True)
        indices = self.reward_buffer.sample(self.batch_size)
        steps = self.reward_buffer.transition[indices]
        rewards = self.reward_buffer.reward[indices].view(-1, 1)
        # Predict r_1 from actual (s_1,b,prev_state')
        r_pred = self.reward_estimator(steps)
        step_loss = F.mse_loss(r_pred, rewards)
        step_loss.backward()
        self.reward_estimator.optim.step()
        
        if traditional_update:
            for target_param, local_param in zip(self.target_reward_estimator.parameters(), self.reward_estimator.parameters()):
                target_param.data.copy_(self.estimator_tau * local_param.data + (1.0 - self.estimator_tau) * target_param.data)
                
        losses['step_loss'] += step_loss.item()
        
        return losses
    
    def _eval_policy(self, problem_instance):
        rewards = []

        best_reward = -np.inf
        training_configs = self.configs_to_consider
        self.configs_to_consider = self.eval_configs
        for _ in range(self.eval_episodes):
            done = False
            action_seq = []
            episode_reward = 0
            state, bridges = self._generate_init_state(problem_instance)
            num_action = len(bridges)
            
            while not done:
                num_action += 1
                action = self._select_action(state)
                reward, next_state, done = self._step(problem_instance, state, action, num_action)
                
                state = next_state
                episode_reward += reward
                transformed_action = self._transform_action(action)
                action_seq += [transformed_action]
                
            rewards.append(episode_reward)
            
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_adaptation = action_seq
        
        self.configs_to_consider = training_configs
        return np.mean(rewards), best_reward, best_adaptation
            
    def _train(self, problem_instance):
        rewards = []
        traditional_td_errors = []
        commutative_td_errors = []
        step_losses = []
        trace_losses = []
        
        best_reward = -np.inf
                
        for episode in range(self.num_episodes):
            done = False
            action_seq = []
            state, bridges = self._generate_init_state(problem_instance)
            num_action = len(bridges)
            
            prev_state = None
            prev_action = None
            prev_reward = None
            
            losses = {'traditional_td_error': 0, 'commutative_td_error': 0, 'step_loss': 0, 'trace_loss': 0}
            while not done:
                num_action += 1
                action = self._select_action(state)
                reward, next_state, done = self._step(problem_instance, state, action, num_action)
                
                if 'approximate' in self.reward_prediction_type:
                    self._add_transition(state, action, reward, next_state, prev_state, prev_action, prev_reward)
                
                losses = self._update_q_table(state, action, reward, next_state, done, losses)

                prev_state = state
                prev_action = action
                prev_reward = reward
                
                state = next_state
                transformed_action = self._transform_action(action)
                action_seq += [transformed_action]
                
            self._decrement_epsilon(episode)

            if 'approximate' in self.reward_prediction_type:
                losses = self._update_estimator(losses)
                
            if episode % self.eval_freq == 0:
                reward, best_eval_reward, best_eval_adaptations = self._eval_policy(problem_instance)
                rewards.append(reward)
                avg_rewards = np.mean(rewards[-self.eval_window:])
                wandb.log({'Average Reward': avg_rewards}, step=episode)
                
                if best_reward > best_eval_reward:
                    best_reward = best_eval_reward
                    best_adaptations = best_eval_adaptations
                
            traditional_td_errors.append(losses['traditional_td_error'])
            commutative_td_errors.append(losses['commutative_td_error'])
            step_losses.append(losses['step_loss'] / (num_action - len(bridges)))
            trace_losses.append(losses['trace_loss'] / (num_action - len(bridges)))
            
            avg_traditional_td_errors = np.mean(traditional_td_errors[-self.sma_window:])
            avg_commutative_td_errors = np.mean(commutative_td_errors[-self.sma_window:])
            avg_step_loss = np.mean(step_losses[-self.sma_window:])
            avg_trace_loss = np.mean(trace_losses[-self.sma_window:])
            
            wandb.log({
                "Average Traditional TD Error": avg_traditional_td_errors,
                "Average Commutative TD Error": avg_commutative_td_errors,
                "Average Step Loss": avg_step_loss, 
                "Average Trace Loss": avg_trace_loss}, step=episode)
            
        return best_reward, best_adaptations
                        
    def _generate_adaptations(self, problem_instance):
        self.epsilon = self.epsilon_start
        
        step_dims = self.state_dims*2 + 1
        
        self.reward_buffer = RewardBuffer(self.batch_size, step_dims, self.rng)
        self.commutative_reward_buffer = CommutativeRewardBuffer(self.batch_size, step_dims, self.rng)
        self.reward_estimator = RewardEstimator(step_dims, self.estimator_alpha, self.dropout_rate)
        self.target_reward_estimator = copy.deepcopy(self.reward_estimator)  
        
        self.reward_estimator.train()
        self.target_reward_estimator.eval()
        
        self.q_table = np.zeros((self.nS, self.action_dims))

        self._init_mapping(problem_instance)
        self._init_wandb(problem_instance)
        
        best_reward, best_adaptation = self._train(problem_instance)
        
        wandb.log({'Adaptation': best_adaptation, 'Final Reward': best_reward})
        wandb.finish()
        
        return best_adaptation
    
    
class CommutativeQTable(BasicQTable):
    def __init__(self, env, random_state, reward_prediction_type, rng):
        super(CommutativeQTable, self).__init__(env, random_state, reward_prediction_type, rng)   
    
    def _update_estimator(self, losses):
        losses = super()._update_estimator(losses, traditional_update=False)
        
        if self.commutative_reward_buffer.real_size < self.batch_size:
            return losses
        
        self.reward_estimator.optim.zero_grad(set_to_none=True)
        commutative_indices = self.commutative_reward_buffer.sample(self.batch_size)
        commutative_steps = self.commutative_reward_buffer.transition[commutative_indices]
        commutative_rewards = self.commutative_reward_buffer.reward[commutative_indices]
        # Approximate r_2 from (s,b,commutative_state) and r_3 from (commutative_state,a,s')
        # MSE Loss: r_2 + r_3 = r_0 + r_1
        summed_r0r1 = torch.sum(commutative_rewards, axis=1).view(-1, 1)
        # Predict r_2 and r_3 from (s,b,commutative_state) and (commutative_state,a,s') respectively
        r2_pred = self.reward_estimator(commutative_steps[:, 0])
        r3_pred = self.reward_estimator(commutative_steps[:, 1])
        trace_loss_r2 = F.mse_loss(r2_pred + r3_pred.detach(), summed_r0r1)
        trace_loss_r3 = F.mse_loss(r2_pred.detach() + r3_pred, summed_r0r1)
        combined_loss = trace_loss_r2 + trace_loss_r3
        combined_loss.backward()
        self.reward_estimator.optim.step()
        
        for target_param, local_param in zip(self.target_reward_estimator.parameters(), self.reward_estimator.parameters()):
            target_param.data.copy_(self.estimator_tau * local_param.data + (1.0 - self.estimator_tau) * target_param.data)
            
        losses['trace_loss'] += trace_loss_r2.item()
        
        return losses
        
    """
    Update Rule 0: Traditional Q-Update
    Q(s_1, b) = Q(s_1, b) + alpha * (r_1 + * max_a Q(s', a) - Q(s_1, b))
    Update Rule 1: Commutative Q-Update
    Q(s_2, a) = Q(s_2, a) + alpha * (r_0 - r_2 + r_1 + max_a Q(s', a) - Q(s_2, a))
    """
    def _update_q_table(self, state, action, reward, next_state, done, losses):
        losses = super()._update_q_table(state, action, reward, next_state, done, losses)
        
        state_idx = self._get_state_idx(state)
        self.ptr_lst[(state_idx, action)] = [reward, next_state]
        
        if self.previous_sample is None:
            self.previous_sample = (state, action, reward)
        else:
            commutative_state = None
            commutative_reward = None
            
            prev_state, prev_action, prev_reward = self.previous_sample
            prev_state_idx = self._get_state_idx(prev_state)
            
            if 'lookup' in self.reward_prediction_type:
                if (prev_state_idx, action) in self.ptr_lst:
                    lookup_reward, commutative_state = self.ptr_lst[(prev_state_idx, action)]      
                    commutative_reward = prev_reward + reward - lookup_reward
            else:            
                commutative_reward = 0 # Placeholder since commutative_reward is predicted by the reward estimator
                transformed_action = self._transform_action(action)
                commutative_state = self._place_bridge(prev_state, transformed_action)
                
                state_idx = self._get_state_idx(state)
                next_state_idx = self._get_state_idx(next_state)    
                
                action_a_success = prev_state_idx != state_idx
                action_b_success = state_idx != next_state_idx                    
                
                if action_a_success and action_b_success:
                    pass
                elif not action_a_success and action_b_success:
                    next_state = commutative_state
                elif action_a_success and not action_b_success:
                    commutative_state = prev_state
                    next_state = state
                else:
                    commutative_state = prev_state
                    next_state = prev_state
                                    
            if commutative_reward is not None:
                losses = super()._update_q_table(commutative_state, prev_action, commutative_reward, next_state, done, losses, traditional_update=False)      
                
            self.previous_sample = (state, action, reward)
            
        if done:
            self.previous_sample = None
            
        return losses
            
    def _generate_adaptations(self, problem_instance):
        self.ptr_lst = {}
        self.previous_sample = None  
        
        return super()._generate_adaptations(problem_instance)
    

class HallucinatedQTable(BasicQTable):
    def __init__(self, env, random_state, rng):
        super(HallucinatedQTable, self).__init__(env, random_state, None, rng)
                
    def _sample_permutations(self, action_seq):
        permutations = {}
        permutations[tuple(action_seq)] = None
        
        tmp_action_seq = action_seq.copy()
        has_terminating_action = self.action_dims - 1 in tmp_action_seq
        terminating_action = tmp_action_seq.pop() if has_terminating_action else None
        
        if len(tmp_action_seq) <= self.max_seq_len:
            tmp_permutations = itertools.permutations(tmp_action_seq)
            for permutation in tmp_permutations:
                if terminating_action is not None:
                    permutations[tuple(permutation) + (terminating_action,)] = None
                else:
                    permutations[tuple(permutation)] = None
        else:
            i = 1
            max_samples = math.factorial(self.max_seq_len)
            while i < max_samples:
                self.rng.shuffle(tmp_action_seq)
                if terminating_action is not None:
                    permutations[tuple(tmp_action_seq) + (terminating_action,)] = None
                else:
                    permutations[tuple(tmp_action_seq)] = None
                i += 1   
        
        return list(permutations.keys())       
    
    # Hallucinate episodes by permuting the action sequence to simulate commutativity
    def _hallucinate(self, start_state, action_seq, episode_reward, losses):
        permutations = self._sample_permutations(action_seq)
        for permutation in permutations:
            num_action = 0
            state = start_state
            terminating_action = permutation[-1]
            for action in permutation:
                num_action += 1
                next_state = self._get_next_state(state, action)
                
                if action == terminating_action:
                    reward = episode_reward
                    done = True
                else:
                    reward = 0
                    done = False
                    
                losses = self._update_q_table(state, action, reward, next_state, done, losses)
                state = next_state
                
        return losses
    
    def _train(self, problem_instance):
        rewards = []
        traditional_td_errors = []
        
        q_table_ckpt = None
        best_avg_rewards = -np.inf
        
        for episode in range(self.num_episodes):
            done = False
            action_seq = []
            episode_reward = 0
            start_state, bridges = self._generate_init_state(problem_instance)
            num_action = len(bridges)
            
            state = start_state
            
            losses = {'traditional_td_error': 0}
            while not done:
                num_action += 1
                action = self._select_action(state)
                reward, next_state, done = self._step(problem_instance, state, action, num_action)   
                             
                state = next_state
                episode_reward += reward
                transformed_action = self._transform_action(action)
                action_seq += [transformed_action]
                
            losses = self._hallucinate(start_state, action_seq, episode_reward, losses)
            
            self._decrement_epsilon(episode)
            
            rewards.append(episode_reward)
            traditional_td_errors.append(losses['traditional_td_error'])

            avg_rewards = np.mean(rewards[-self.sma_window:])
            avg_traditional_td_errors = np.mean(traditional_td_errors[-self.sma_window:])

            wandb.log({
                "Average Reward": avg_rewards,
                "Average Traditional TD Error": avg_traditional_td_errors}, step=episode)
            
            if episode > 20000 and avg_rewards > best_avg_rewards:
                best_avg_rewards = avg_rewards
                q_table_ckpt = copy.deepcopy(self.q_table)
            
        return q_table_ckpt
    