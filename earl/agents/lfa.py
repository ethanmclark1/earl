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


# Linear Function Approximation
class BasicLFA(EA):
    def __init__(self, env, random_state, reward_prediction_type, rng):
        super(BasicLFA, self).__init__(env, random_state, rng)
        
        self.weights = None
        self.reward_estimator = None
        self.target_reward_estimator = None
        self.reward_buffer = None
        self.comm_reward_buffer = None
        self.reward_prediction_type = reward_prediction_type
    
        self.num_features = (2 * self.state_dims) + self.action_dims
        
    def _init_hyperparams(self):
        self.eval_episodes = 25
        self.warmup_episodes = 2000
        
        # Reward Estimator
        self.gamma = 0.50
        self.batch_size = 128
        self.step_size = 250000
        self.dropout_rate = 0.50
        self.estimator_tau = 0.25
        self.estimator_alpha = 0.0003
        self.model_save_interval = 2500
        
        # LFA
        self.alpha = 0.001
        self.max_seq_len = 7
        self.sma_window = 250
        self.epsilon_start = 1
        self.min_epsilon = 0.10
        self.num_episodes = 10000
        self.epsilon_decay = 0.00005 if self.random_state else 0.00001

    def _init_wandb(self, problem_instance):
        config = super()._init_wandb(problem_instance)
        config.alpha = self.alpha
        config.gamma = self.gamma
        config.step_size = self.step_size
        config.batch_size = self.batch_size
        config.sma_window = self.sma_window
        config.max_action = self.max_action
        config.batch_size = self.batch_size
        config.action_cost = self.action_cost
        config.min_epsilon = self.min_epsilon
        config.max_seq_len = self.max_seq_len
        config.random_state = self.random_state
        config.num_episodes = self.num_episodes
        config.dropout_rate = self.dropout_rate
        config.epsilon_decay = self.epsilon_decay
        config.estimator_tau = self.estimator_tau
        config.percent_holes = self.percent_holes
        config.warmup_episodes = self.warmup_episodes
        config.estimator_alpha = self.estimator_alpha
        config.reward_estimator = self.reward_estimator 
        config.model_save_interval = self.model_save_interval
        config.action_success_rate = self.action_success_rate
        config.configs_to_consider = self.configs_to_consider
        config.reward_prediction_type = self.reward_prediction_type
        
    def _save_estimator(self, problem_instance, episode):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        if episode % self.model_save_interval == 0:
            filename = f'{problem_instance}_{episode}.pt'
            file_path = os.path.join(self.output_dir, filename)
            torch.save(self.reward_estimator.state_dict(), file_path)
        
    def _decrement_epsilon(self, episode):
        if 'approximate' not in self.reward_prediction_type or episode > self.warmup_episodes:
            self.epsilon -= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.min_epsilon)
        
    def _select_action(self, state):
        if self.rng.random() < self.epsilon:
            action = self.rng.integers(self.action_dims)
        else:
            action, _ = self._get_max_q_value(state)
            
        return action
    
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
    
    # TODO: Consider stochasticity and how it effects the for loop
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
    
    # TODO: Look more into this
    def _update_weights(self, state, action, reward, next_state, done, losses, traditional_update=True):
        if 'approximate' in self.reward_prediction_type:
            with torch.no_grad():
                state_proxy = self._get_mutable_state(state)
                next_state_proxy = self._get_mutable_state(next_state)
                
                state_tensor = torch.as_tensor(state_proxy, dtype=torch.float)
                action_tensor = torch.as_tensor([action], dtype=torch.float)
                next_state_tensor = torch.as_tensor(next_state_proxy, dtype=torch.float)
                
                step = torch.cat([state_tensor, action_tensor, next_state_tensor], dim=-1)
                reward = self.target_reward_estimator(step).item()
            
        features = self._get_features(state, action)
        current_q_value = np.dot(self.weights, features)
        _, max_next_q_value = self._get_max_q_value(next_state)
        
        td_target = reward + (1 - done) * max_next_q_value
        td_error = td_target - current_q_value

        self.weights += self.alpha * td_error * features
        
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
        self.reward_estimator.scheduler.step()
        
        if traditional_update:
            for target_param, local_param in zip(self.target_reward_estimator.parameters(), self.reward_estimator.parameters()):
                target_param.data.copy_(self.estimator_tau * local_param.data + (1.0 - self.estimator_tau) * target_param.data)
                
        losses['step_loss'] += step_loss.item()
        
        return losses

    def _train(self, problem_instance):
        rewards = []
        traditional_td_errors = []
        commutative_td_errors = []
        step_losses = []
        trace_losses = []
        
        weights_ckpt = None
        best_avg_rewards = -np.inf
        
        for episode in range(self.num_episodes):
            done = False
            action_seq = []
            episode_reward = 0
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
                
                losses = self._update_weights(state, action, reward, next_state, done, losses)

                prev_state = state
                prev_action = action
                prev_reward = reward
                
                state = next_state
                episode_reward += reward
                transformed_action = self._transform_action(action)
                action_seq += [transformed_action]
                
            self._decrement_epsilon(episode)

            if 'approximate' in self.reward_prediction_type:
                losses = self._update_estimator(losses)

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
            
            if episode > 2000 and avg_rewards > best_avg_rewards:
                best_avg_rewards = avg_rewards
                weights_ckpt = copy.deepcopy(self.q_table)
                
            self._save_estimator(problem_instance, episode)
            
        return weights_ckpt
    
    def _get_best_adaptation(self, problem_instance, weights_ckpt):
        best_rewards = -np.inf
        best_adaptation = None
        
        self.epsilon = 0
        self.weights = weights_ckpt
        self.configs_to_consider = 100
        for _ in range(self.eval_episodes):
            done = False
            action_seq = []
            episode_reward = 0
            state, adaptations = self._generate_fixed_state(problem_instance)
            num_action = len(adaptations)
            
            while not done:
                num_action += 1
                action = self._select_action(state)
                reward, next_state, done = self._step(problem_instance, state, action, num_action)
                
                state = next_state
                episode_reward += reward
                transformed_action = self._transform_action(action)
                action_seq += [transformed_action]
                
            if episode_reward > best_rewards:
                best_rewards = episode_reward
                best_adaptation = action_seq
        
        return best_adaptation, best_rewards
            
    def _generate_adaptations(self, problem_instance):
        self.epsilon = self.epsilon_start
        
        step_dims = self.state_dims*2 + 1
        
        self.reward_buffer = RewardBuffer(self.batch_size, step_dims, self.rng)
        self.commutative_reward_buffer = CommutativeRewardBuffer(self.batch_size, step_dims, self.rng)
        self.reward_estimator = RewardEstimator(step_dims, self.estimator_alpha, self.step_size, self.gamma, self.dropout_rate)
        self.target_reward_estimator = copy.deepcopy(self.reward_estimator)  
        
        self.reward_estimator.train()
        self.target_reward_estimator.eval()
        
        self.weights = np.zeros(self.num_features)
        
        self._init_mapping(problem_instance)
        self._init_wandb(problem_instance)
        
        weights_ckpt = self._train(problem_instance)
        best_adaptation, best_reward = self._get_best_adaptation(problem_instance, weights_ckpt)
        
        wandb.log({'Adaptation': best_adaptation, 'Final Reward': best_reward})
        wandb.finish()
        
        return best_adaptation
    
    
class CommutativeLFA(BasicLFA):
    def __init__(self, env, random_state, reward_prediction_type, rng):
        super(CommutativeLFA, self).__init__(env, random_state, reward_prediction_type, rng) 
        
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
    def _update_weights(self, state, action, reward, next_state, done, losses):
        losses = super()._update_weights(state, action, reward, next_state, done, losses)
        
        if self.reward_prediction_type not in ['lookup', 'approximate w/ commutative update']:
            return losses
        
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
                
                state_idx = self._get_state_idx(state)
                next_state_idx = self._get_state_idx(next_state)    
                
                action_b_success = state_idx != next_state_idx
                
                # If action b is successful, then update (commutative_state, a) -> next_state
                if action_b_success:
                    transformed_action = self._transform_action(action)
                    commutative_state = self._place_bridge(prev_state, transformed_action)
                # If action b fails, then we don't know commutative_state so update (prev_state, a) -> state
                else:
                    commutative_state = prev_state
                    next_state = state
                                    
            if commutative_reward is not None:
                losses = super()._update_weights(commutative_state, prev_action, commutative_reward, next_state, done, losses, traditional_update=False)      
                
            self.previous_sample = (state, action, reward)
            
        if done:
            self.previous_sample = None
            
        return losses
            
    def _generate_adaptations(self, problem_instance):
        self.ptr_lst = {}
        self.previous_sample = None  
        
        return super()._generate_adaptations(problem_instance)
    

class HallucinatedLFA(BasicLFA):
    def __init__(self, env, random_state, rng):
        super(HallucinatedLFA, self).__init__(env, random_state, None, rng)
        
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
                    
                losses = self._update_weights(state, action, reward, next_state, done, losses)
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
            
            if episode > 2000 and avg_rewards > best_avg_rewards:
                best_avg_rewards = avg_rewards
                q_table_ckpt = copy.deepcopy(self.q_table)
            
        return q_table_ckpt