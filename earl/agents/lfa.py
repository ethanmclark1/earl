import math
import wandb
import torch
import random
import itertools
import numpy as np
import torch.nn.functional as F

from agents.utils.ea import EA
from agents.utils.networks import RewardEstimator


# Linear Function Approximation
class BasicLFA(EA):
    def __init__(self, env, rng, random_state, reward_prediction_type=None):
        super(BasicLFA, self).__init__(env, rng, random_state)
        self.random_state = random_state
        
        self.weights = None
        # Add a dummy action (+1) to terminate the episode
        
        self.alpha = 0.001
        self.max_seq_len = 7
        self.epsilon_start = 1
        self.sma_window = 1000
        self.min_epsilon = 0.10
        self.num_episodes = 10000
        self.estimator_alpha = 0.003
        self.reward_prediction_type = reward_prediction_type

    def _init_wandb(self, problem_instance):
        config = super()._init_wandb(problem_instance)
        config.alpha = self.alpha
        config.sma_window = self.sma_window
        config.max_action = self.max_action
        config.min_epsilon = self.min_epsilon
        config.action_cost = self.action_cost
        config.max_seq_len = self.max_seq_len
        config.random_state = self.random_state
        config.num_episodes = self.num_episodes
        config.epsilon_decay = self.epsilon_decay
        config.percent_holes = self.percent_holes
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
            original_action, _ = self._get_max_q_value(state)
            
        transformed_action = self._transform_action(original_action)
        return transformed_action, original_action
    
    # Feature vector is a one hot encoding of the state and the action
    def _get_features(self, state, action):
        state_features = np.empty(2 * self.state_dims)
        
        mutable_cells = list(map(lambda x: (x // self.num_cols, x % self.num_cols), self.mapping.values()))
        rows, cols = zip(*mutable_cells)
        mutable_state = state[rows, cols]
        
        for i, cell in enumerate(mutable_state):
            if cell == 1:
                state_features[2 * i] = 1
                state_features[2 * i + 1] = 0
            else:
                state_features[2 * i] = 0 
                state_features[2 * i + 1] = 1
                
        action_features = np.zeros(self.action_dims)
        action_features[action] = 1
        
        return np.concatenate([state_features, action_features])
        
    def _get_max_q_value(self, state):
        best_action = None
        max_q_value = -np.inf
        
        for action in range(self.action_dims):
            features = self._get_features(state, action)
            q_value = np.dot(self.weights, features)
            
            if q_value > max_q_value:
                best_action = action
                max_q_value = q_value
        
        return best_action, max_q_value
    
    def _update_weights(self, state, action, reward, next_state, done, losses, episode=None, traditional_update=True):
        features = self._get_features(state, action)
        current_q_value = np.dot(self.weights, features)
        
        # Use reward estimator to approximate reward for traditional Q-Update 
        # if traditional_update and self.reward_prediction_type == 'approximate':
        #     step = torch.FloatTensor([state, action, next_state])
        #     reward = self.reward_estimator(step)
            
        _, max_next_q_value = self._get_max_q_value(next_state)
        td_target = reward + (1 - done) * max_next_q_value
        td_error = td_target - current_q_value

        self.weights += self.alpha * td_error * features
        
        if traditional_update:
            losses['traditional_td_error'] += abs(td_error)
        else:
            losses['commutative_td_error'] += abs(td_error)
            
        return losses

    def _train(self, problem_instance):
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
                
                losses = self._update_weights(state, original_action, reward, next_state, done, losses, episode)
                
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
            
    def _generate_adaptations(self, problem_instance):
        self.epsilon = self.epsilon_start
        
        self._init_problem(problem_instance)
        self._init_wandb(problem_instance)
        
        # Add a dummy action (+1) to terminate the episode
        self.action_dims = self.state_dims + 1 
        self.num_features = (2 * self.state_dims) + self.action_dims
        self.weights = np.zeros(self.num_features)
        
        best_adaptation, best_reward = self._train(problem_instance)
        
        wandb.log({'Adaptation': best_adaptation, 'Final Reward': best_reward})
        wandb.finish()
        
        return best_adaptation
    
    
class CommutativeLFA(BasicLFA):
    def __init__(self, env, rng, random_state, reward_prediction_type):
        super(CommutativeLFA, self).__init__(env, rng, random_state, reward_prediction_type) 
        self.reward_estimator = RewardEstimator(self.estimator_alpha)
        
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
        
        return traces[3], step_loss.item(), trace_loss_r2.item()
    
    def _update_weights(self, state, action, reward, next_state, done, losses, episode):
        losses = super()._update_weights(state, action, reward, next_state, done, losses, episode)
        
        state_idx = self._get_state_idx(state)
        self.ptr_lst[(state_idx, action)] = (reward, next_state)
        
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
                r3_pred = self.reward_estimator(r3_step).item()
                
                losses['step_loss'] += step_loss
                losses['trace_loss'] += trace_loss
                
            if r3_pred is not None:
                losses = super()._update_weights(s_2, a, r3_pred, s_prime, done, losses, episode, traditional_update=False)      

            self.previous_sample = (state, action, reward)
    
        if done:
            self.previous_sample = None
            
    def _generate_adaptations(self, problem_instance):
        self.ptr_lst = {}
        self.previous_sample = None
        
        return super()._generate_adaptations(problem_instance)
    

class HallucinatedLFA(BasicLFA):
    def __init__(self, env, rng, random_state):
        super(HallucinatedLFA, self).__init__(env, rng, random_state)
        
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
                random.shuffle(tmp_action_seq)
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
                    
                self._update_weights(state, original_action, reward, next_state, done)
                state = next_state
    
    def _train(self, problem_instance):
        rewards = []
        
        best_reward = -np.inf
        best_action_seq = None
        
        for _ in range(self.num_episodes):
            done = False
            num_action = 0
            action_seq = []
            episode_reward = 0
            start_state = self._get_state(problem_instance)
            state = start_state
            while not done:
                num_action += 1
                transformed_action, original_action = self._select_action(state)
                reward, next_state, done = self._step(problem_instance, state, transformed_action, num_action)   
                             
                state = next_state
                action_seq += [original_action]
                episode_reward += reward
                
            self._hallucinate(start_state, action_seq, episode_reward)
            self.epsilon *= self.epsilon_decay
            
            rewards.append(episode_reward)
            avg_rewards = np.mean(rewards[-self.sma_window:])
            wandb.log({"Average Reward": avg_rewards}) 
            
            if episode_reward > best_reward:
                best_action_seq = action_seq
                best_reward = episode_reward
            
        return best_action_seq, best_reward