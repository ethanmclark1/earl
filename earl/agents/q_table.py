import math
import copy
import wandb
import torch
import itertools
import numpy as np
import torch.nn.functional as F

from agents.utils.ea import EA
from agents.utils.networks import RewardEstimator


class BasicQTable(EA):
    def __init__(self, env, rng, is_online, random_state, reward_prediction_type=None):
        super(BasicQTable, self).__init__(env, rng, random_state)
                
        self.q_table = None
        self.reward_estimator = None
        self.nS = 2 ** self.state_dims

        self.gamma = 0.95
        self.alpha = 0.0005
        self.step_size = 100
        self.max_seq_len = 7
        self.epsilon_start = 1
        self.sma_window = 2500
        self.min_epsilon = 0.05
        self.eval_episodes = 500
        self.num_episodes = 300000
        self.is_online = is_online
        self.estimator_tau = 0.05
        self.sma_window_eval = 500
        self.estimator_alpha = 0.01
        self.offline_episodes = 400000
        self.reward_prediction_type = reward_prediction_type
        self.epsilon_decay = 0.00005 if self.random_state else 0.00001
        
    def _init_wandb(self, problem_instance):
        config = super()._init_wandb(problem_instance)
        config.alpha = self.alpha
        config.gamma = self.gamma
        config.is_online = self.is_online
        config.step_size = self.step_size
        config.sma_window = self.sma_window
        config.max_action = self.max_action
        config.action_cost = self.action_cost
        config.min_epsilon = self.min_epsilon
        config.max_seq_len = self.max_seq_len
        config.random_state = self.random_state
        config.num_episodes = self.num_episodes
        config.eval_episodes = self.eval_episodes
        config.epsilon_decay = self.epsilon_decay
        config.percent_holes = self.percent_holes
        config.estimator_tau = self.estimator_tau
        config.estimator_alpha = self.estimator_alpha
        config.sma_window_eval = self.sma_window_eval
        config.reward_estimator = self.reward_estimator 
        config.offline_episodes = self.offline_episodes
        config.action_success_rate = self.action_success_rate
        config.configs_to_consider = self.configs_to_consider
        config.reward_prediction_type = self.reward_prediction_type
    
    def _decrement_epsilon(self):
        self.epsilon -= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.min_epsilon)
        
    def _select_action(self, state):
        if self.rng.random() < self.epsilon:
            original_action = self.rng.integers(self.action_dims)
        else:
            state_idx = self._get_state_idx(state)
            original_action = self.q_table[state_idx].argmax()
            
        transformed_action = self._transform_action(original_action)
        return transformed_action, original_action
    
    def _update_q_table(self, state, action, reward, next_state, done, losses, episode=None, traditional_update=True):
        state = self._get_state_idx(state)
        next_state = self._get_state_idx(next_state)
        
        # Use reward estimator to approximate reward for traditional Q-Update 
        if traditional_update and self.reward_prediction_type == 'approximate':
            step = torch.FloatTensor([state, action, next_state])
            reward = self.target_reward_estimator(step).item()
        
        td_target = reward + (1 - done) * self.q_table[next_state].max() 
        td_error = td_target - self.q_table[state, action]
        
        self.q_table[state, action] += self.alpha * td_error
        
        if traditional_update:
            losses['traditional_td_error'] += abs(td_error)
        else:
            losses['commutative_td_error'] += abs(td_error)
            
        return losses
            
    def _online_train(self, problem_instance):
        rewards = []
        traditional_td_errors = []
        commutative_td_errors = []
        step_losses = []
        trace_losses = []
        
        best_reward = -np.inf
        best_action_seq = None
        
        for episode in range(self.num_episodes):
            done = False
            action_seq = []
            episode_reward = 0
            state, bridges = self._generate_init_state(problem_instance)
            num_action = len(bridges)
            losses = {'traditional_td_error': 0, 'commutative_td_error': 0, 'step_loss': 0, 'trace_loss': 0}
            while not done:
                num_action += 1
                transformed_action, original_action = self._select_action(state)
                reward, next_state, done = self._step(problem_instance, state, transformed_action, num_action)
                
                losses = self._update_q_table(state, original_action, reward, next_state, done, losses, episode)

                state = next_state
                episode_reward += reward
                action_seq += [original_action]
                
            self._decrement_epsilon()
            
            rewards.append(episode_reward)
            traditional_td_errors.append(losses['traditional_td_error'])
            commutative_td_errors.append(losses['commutative_td_error'])
            step_losses.append(losses['step_loss'] / (num_action - len(bridges)))
            trace_losses.append(losses['trace_loss'] / (num_action - len(bridges)))
            
            avg_rewards = np.mean(rewards[-self.sma_window:])
            avg_traditional_td_errors = np.mean(traditional_td_errors[-self.sma_window:])
            avg_commutative_td_errors = np.mean(commutative_td_errors[-self.sma_window:])
            avg_step_loss = np.mean(step_losses[-self.sma_window:])
            avg_trace_loss = np.mean(trace_losses[-self.sma_window:])
            
            wandb.log({
                "Average Reward": avg_rewards,
                "Average Traditional TD Error": avg_traditional_td_errors,
                "Average Commutative TD Error": avg_commutative_td_errors,
                "Average Step Loss": avg_step_loss, 
                "Average Trace Loss": avg_trace_loss}, step=episode)
            
            if episode_reward > best_reward:
                best_action_seq = action_seq
                best_reward = episode_reward
            
        return best_action_seq, best_reward
    
    def _offline_train(self, traces):
        non_empty_rows = ~np.all(traces == 0, axis=1)
        filtered_traces = traces[non_empty_rows]
        
        episode = 0
        num_action = 0
        losses = {'traditional_td_error': 0, 'commutative_td_error': 0, 'step_loss': 0, 'trace_loss': 0}
        for trace in filtered_traces:
            num_action += 1
            state_idx, action, reward, next_state_idx, done = trace
            
            state = self._get_state_from_idx(int(state_idx))
            action = int(action)
            next_state = self._get_state_from_idx(int(next_state_idx))
            done = bool(done)
            
            self._update_q_table(state, action, reward, next_state, done, losses, episode)
            
            if done:                
                episode += 1
                num_action = 0
    
    def _evaluate(self, problem_instance):
        rewards = []
        best_reward = -np.inf
        best_action_seq = None
        
        self.epsilon = 0
        for episode in range(self.eval_episodes):
            done = False
            action_seq = []
            episode_reward = 0
            state, bridges = self._generate_init_state(problem_instance)
            num_action = len(bridges)
            while not done:
                num_action += 1
                transformed_action, original_action = self._select_action(state)
                reward, next_state, done = self._step(problem_instance, state, transformed_action, num_action)
                
                state = next_state
                episode_reward += reward
                action_seq += [original_action]
                
            rewards.append(episode_reward)
            avg_reward = np.mean(rewards[-self.sma_window_eval:])
            wandb.log({"Average Reward": avg_reward}, step=episode)
            
            if episode_reward > best_reward:
                best_action_seq = action_seq
                best_reward = episode_reward
        
        return best_action_seq, best_reward
            
    def _generate_adaptations(self, problem_instance):
        self.epsilon = self.epsilon_start
        
        self._init_mapping(problem_instance)
        self._init_wandb(problem_instance)
        
        self.q_table = np.zeros((self.nS, self.action_dims))
        
        if self.is_online:
            best_adaptation, best_reward = self._online_train(problem_instance)
        else:
            traces = self._get_traces(problem_instance)
            self._offline_train(traces)
            best_adaptation, best_reward = self._evaluate(problem_instance)
        
        wandb.log({'Adaptation': best_adaptation, 'Final Reward': best_reward})
        wandb.finish()
        
        return best_adaptation
    
    
class CommutativeQTable(BasicQTable):
    def __init__(self, env, rng, is_online, random_state, reward_prediction_type):
        super(CommutativeQTable, self).__init__(env, rng, is_online, random_state, reward_prediction_type)        
    
    def _update_estimator(self, traces, r_0, r_1):
        traces = torch.FloatTensor(traces)
        r_0 = torch.FloatTensor([r_0])
        r_1 = torch.FloatTensor([r_1])
        
        r0r1_pred = self.reward_estimator(traces[:2])
        self.reward_estimator.optim.zero_grad()
        combined_r0r1 = torch.cat([r_0, r_1], dim=-1).view(-1, 1)
        step_loss = F.mse_loss(combined_r0r1, r0r1_pred)
        step_loss.backward(retain_graph=True)
        self.reward_estimator.optim.step()
        
        r2r3_pred = self.reward_estimator(traces[2:])
        self.reward_estimator.optim.zero_grad()
        trace_loss_r2 = F.mse_loss(r_0 + r_1, r2r3_pred[0] + r2r3_pred[1].detach())
        trace_loss_r3 = F.mse_loss(r_0 + r_1, r2r3_pred[0].detach() + r2r3_pred[1])
        combined_loss = trace_loss_r2 + trace_loss_r3
        combined_loss.backward()
        self.reward_estimator.optim.step()
        
        self.reward_estimator.scheduler.step()
        
        for target_param, local_param in zip(self.target_reward_estimator.parameters(), self.reward_estimator.parameters()):
            target_param.data.copy_(self.estimator_tau * local_param.data + (1.0 - self.estimator_tau) * target_param.data)
        
        return traces[3], step_loss.item(), trace_loss_r2.item()
        
    """
    Update Rule 0: Traditional Q-Update
    Q(s_1, b) = Q(s_1, b) + alpha * (r_1 + * max_a Q(s', a) - Q(s_1, b))
    Update Rule 1: Commutative Q-Update
    Q(s_2, a) = Q(s_2, a) + alpha * (r_0 - r_2 + r_1 + max_a Q(s', a) - Q(s_2, a))
    """
    def _update_q_table(self, state, action, reward, next_state, done, losses, episode):
        losses = super()._update_q_table(state, action, reward, next_state, done, losses, episode)
        
        state_idx = self._get_state_idx(state)
        self.ptr_lst[(state_idx, action)] = [reward, next_state]
        
        if self.previous_sample is None:
            self.previous_sample = (state, action, reward)
        else:
            s, a, r_0 = self.previous_sample
            
            s_1 = state
            b = action
            r_1 = reward
            s_prime = next_state
            
            s_2 = None
            r_2 = None
            r3_pred = None
            
            s_idx = self._get_state_idx(s)
            if 'lookup' in self.reward_prediction_type:
                if (s_idx, b) in self.ptr_lst:
                    r_2, s_2 = self.ptr_lst[(s_idx, b)]      
                    r3_pred = r_0 + r_1 - r_2
            else:
                transformed_action = self._transform_action(b)
                s_2 = self._get_next_state(s, transformed_action)
                
                s_1_idx = self._get_state_idx(s_1)
                s_2_idx = self._get_state_idx(s_2)
                s_prime_idx = self._get_state_idx(s_prime)

                traces = np.array([[s_idx, a, s_1_idx], [s_1_idx, b, s_prime_idx], [s_idx, b, s_2_idx], [s_2_idx, a, s_prime_idx]])
                
                r3_step, step_loss, trace_loss = self._update_estimator(traces, r_0, r_1)
                r3_pred = self.target_reward_estimator(r3_step).item()
                
                losses['step_loss'] += step_loss
                losses['trace_loss'] += trace_loss
         
            if r3_pred is not None:
                losses = super()._update_q_table(s_2, a, r3_pred, s_prime, done, losses, episode, traditional_update=False)      
                
            self.previous_sample = (s_1, b, r_1)
            
        if done:
            self.previous_sample = None
            
        return losses
            
    def _generate_adaptations(self, problem_instance):
        self.ptr_lst = {}
        self.previous_sample = None 
        
        self.reward_estimator = RewardEstimator(self.estimator_alpha, self.step_size, self.gamma)
        self.target_reward_estimator = copy.deepcopy(self.reward_estimator)     
        
        return super()._generate_adaptations(problem_instance)
    

class HallucinatedQTable(BasicQTable):
    def __init__(self, env, rng, is_online, random_state):
        super(HallucinatedQTable, self).__init__(env, rng, is_online, random_state)
                
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
    def _hallucinate(self, start_state, action_seq, episode_reward):
        permutations = self._sample_permutations(action_seq)
        for permutation in permutations:
            num_action = 0
            state = start_state
            terminating_action = permutation[-1]
            for original_action in permutation:
                num_action += 1
                transformed_action = self._transform_action(original_action)
                next_state = self._get_next_state(state, transformed_action)
                
                if original_action == terminating_action:
                    reward = episode_reward
                    done = True
                else:
                    reward = 0
                    done = False
                    
                self._update_q_table(state, original_action, reward, next_state, done)
                state = next_state
    
    def _train(self, problem_instance):
        rewards = []
        
        best_reward = -np.inf
        best_action_seq = None
        
        for _ in range(self.num_episodes):
            done = False
            action_seq = []
            episode_reward = 0
            start_state, bridges = self._generate_init_state(problem_instance)
            num_action = len(bridges)
            state = start_state
            while not done:
                num_action += 1
                transformed_action, original_action = self._select_action(problem_instance, state)
                reward, next_state, done = self._step(problem_instance, state, transformed_action, num_action)   
                             
                state = next_state
                action_seq += [original_action]
                episode_reward += reward
                
            self._hallucinate(start_state, action_seq, episode_reward)
            self._decrement_epsilon()
            
            rewards.append(episode_reward)
            avg_rewards = np.mean(rewards[-self.sma_window:])
            wandb.log({"Average Reward": avg_rewards})
            
            if episode_reward > best_reward:
                best_action_seq = action_seq
                best_reward = episode_reward
            
        return best_action_seq, best_reward
    