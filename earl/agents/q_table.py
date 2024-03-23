import math
import copy
import wandb
import itertools
import numpy as np

from agents.utils.ea import EA
from agents.utils.networks import RewardEstimator
from agents.utils.buffer import RewardBuffer, CommutativeRewardBuffer

from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder


class BasicQTable(EA):
    def __init__(self, env, num_instances, random_state, reward_prediction_type):
        super(BasicQTable, self).__init__(env, num_instances, random_state)
        self._init_hyperparams()
        
        self.name = self.__class__.__name__
        self.output_dir = f'earl/agents/history/estimator/{self.name.lower()}'
                
        self.q_table = None
        self.estimator = None
        self.reward_prediction_type = reward_prediction_type
        self.encoder = OneHotEncoder(categories=[range(self.action_dims)], sparse_output=False)

        self.nS = 2 ** self.state_dims
        self.step_dims = 2 * self.state_dims + self.action_dims
        
    def _init_hyperparams(self):        
        # Reward Estimator
        self.batch_size = 64 if self.problem_size == '8x8' else 32
        self.buffer_size = 128 if self.problem_size == '8x8' else 64
        self.estimator_alpha = 0.003 if self.problem_size == '8x8' else 0.08

        # Q-Table
        self.epsilon_start = 1
        self.min_epsilon = 0.10
        self.max_seq_len = 7 if self.problem_size == '8x8' else 5
        self.alpha = 0.0005 if self.problem_size == '8x8' else 0.001
        self.sma_window = 5000 if self.problem_size == '8x8' else 100
        self.num_episodes = 500000 if self.problem_size == '8x8' else 2500
        
        epsilon_decay_values = {
            ('8x8', True): 0.0003,
            ('8x8', False): 0.0001,
            ('4x4', True): 0.0010,
            ('4x4', False): 0.025,
        }
        self.epsilon_decay = epsilon_decay_values.get((self.problem_size, self.random_state), None)
        
        # Evaluation Settings
        self.convergence_window = 100
        self.variance_threshold = 1.5
        self.difference_threshold = 1.25
        self.eval_episodes = 25
        self.eval_freq = 500 if self.problem_size == '8x8' else 25
        self.eval_window = 40 if self.problem_size == '8x8' else 25
        self.eval_configs = 150 if self.problem_size == '8x8' else 75
        
    def _init_wandb(self, problem_instance):
        config = super()._init_wandb(problem_instance)
        config.alpha = self.alpha
        config.eval_freq = self.eval_freq
        config.sma_window = self.sma_window
        config.max_action = self.max_action
        config.num_bridges = self.state_dims
        config.eval_window = self.eval_window
        config.action_cost = self.action_cost
        config.min_epsilon = self.min_epsilon
        config.max_seq_len = self.max_seq_len
        config.random_state = self.random_state
        config.num_episodes = self.num_episodes
        config.eval_configs = self.eval_configs
        config.problem_size = self.problem_size
        config.epsilon_decay = self.epsilon_decay
        config.eval_episodes = self.eval_episodes
        config.warmup_episodes = self.warmup_episodes
        config.estimator_alpha = self.estimator_alpha
        config.convergence_window = self.convergence_window
        config.variance_threshold = self.variance_threshold
        config.action_success_rate = self.action_success_rate
        config.configs_to_consider = self.configs_to_consider
        config.difference_threshold = self.difference_threshold
        config.reward_prediction_type = self.reward_prediction_type
    
    # Only decrement epsilon after reward estimator has been sufficiently trained
    def _decrement_epsilon(self, episode):
        if self.reward_prediction_type != 'approximate' or episode >= self.warmup_episodes:
            self.epsilon -= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.min_epsilon)
            
    def _select_action(self, state, is_train=True):
        if is_train and self.rng.random() < self.epsilon:
            action = self.rng.integers(self.action_dims)
        else:
            state_proxy = self._get_state_proxy(state)
            action = max(self.q_table[tuple(state_proxy)], key=self.q_table[tuple(state_proxy)].get)
            
        return action
    
    # def _add_transition(self, state_proxy, action, reward, next_state_proxy, prev_state_proxy, prev_action, prev_reward):        
    #     self.reward_buffer.add(state_proxy, action, reward, next_state_proxy)
        
    #     if 'Commutative' in self.name and action != 0 and prev_state_proxy is not None:
    #         commutative_state_proxy, next_state_proxy = self._reassign_states(prev_state_proxy, action, state_proxy, next_state_proxy)
    #         self.commutative_reward_buffer.add(prev_state_proxy, action, prev_reward, commutative_state_proxy, prev_action, reward, next_state_proxy)
    
    # def _update_estimator(self, losses):
    #     if self.batch_size > self.reward_buffer.real_size:
    #         return losses
        
    #     idxs = self.reward_buffer.sample(self.batch_size)
    #     transitions = self.reward_buffer.transition[idxs]
    #     rewards = self.reward_buffer.reward[idxs]
        
    #     prediction = self.estimator.predict(transitions)
    #     error = rewards - prediction
        
    #     gradients_weights = -error.reshape(-1, 1) * transitions
        
    #     gradients = {
    #         'weights': gradients_weights.mean(axis=0),
    #         'bias': -error.mean()
    #     }
        
    #     self.estimator.update_weights(gradients)
                
    #     losses['step_loss'] += (error ** 2).mean()
        
    #     if 'Commutative' in self.name and self.commutative_reward_buffer.real_size >= self.batch_size:
    #         idxs = self.commutative_reward_buffer.sample(self.batch_size)
    #         transitions = self.commutative_reward_buffer.transition[idxs]
    #         rewards = self.commutative_reward_buffer.reward[idxs]
            
    #         r2_pred = self.estimator.predict(transitions[:, 0])
    #         r3_pred = self.estimator.predict(transitions[:, 1])
    #         error = (rewards[:, 0] + rewards[:, 1]) - (r2_pred + r3_pred)
            
    #         r2_gradient_weights = -error.reshape(-1, 1) * transitions[:, 0]
    #         r3_gradient_weights = -error.reshape(-1, 1) * transitions[:, 1]

    #         gradients = {
    #             'weights': (r2_gradient_weights + r3_gradient_weights).mean(axis=0),
    #             'bias': -error.mean()
    #         }
    #         self.estimator.update_weights(gradients)
            
    #         losses['trace_loss'] += (error ** 2).mean()
        
    #     return losses        
                
    def _update_q_table(self, state_proxy, action, reward, next_state_proxy, done, losses, traditional_update=True, num_action=None):
        # if self.reward_prediction_type == 'approximate':
        #     action_reshaped = np.array([action]).reshape(-1, 1)
        #     action_enc = self.encoder.fit_transform(action_reshaped).reshape(-1).astype(int)
        #     features = np.concatenate([state_proxy, action_enc, next_state_proxy]).reshape(1, -1)
                
        #     reward = self.estimator.predict(features).item()
        
        td_target = reward + (1 - done) * max(self.q_table[tuple(next_state_proxy)].values())
        td_error = td_target - self.q_table[tuple(state_proxy)][action]
        
        self.q_table[tuple(state_proxy)][action] += self.alpha * td_error
        
        if traditional_update:
            self.traditional_update += 1
            losses['traditional_td_error'] += abs(td_error)
        else:
            self.commutative_update += 1
            losses['commutative_td_error'] += abs(td_error)
            
        return losses
            
    def _eval_policy(self,):
        returns = []

        training_configs = self.configs_to_consider
        self.configs_to_consider = self.eval_configs
        for _ in range(self.eval_episodes):
            done = False
            bridges = []
            episode_reward = 0
            state, adaptations = self._generate_fixed_state()
            num_action = len(adaptations)
            
            while not done:
                num_action += 1
                action = self._select_action(state, is_train=False)
                reward, next_state, done = self._step(state, action, num_action)
                
                state = next_state
                episode_reward += reward
                transformed_action = self._transform_action(action)
                bridges += [transformed_action]
                
            returns.append(episode_reward)
        
        self.configs_to_consider = training_configs
        
        avg_return = np.mean(returns)
        return avg_return, bridges
            
    def _train(self):
        eval_returns = []
        traditional_td_errors = []
        commutative_td_errors = []
        step_losses = []
        trace_losses = []
        
        best_return = -np.inf
                
        for episode in range(self.num_episodes):
            done = False
            state, adaptations = self._generate_start_state()
            state_proxy = self._get_state_proxy(state)
            num_action = len(adaptations)
            
            # prev_state_proxy = None
            # prev_action = None
            # prev_reward = None
            
            losses = {'traditional_td_error': 0, 'commutative_td_error': 0, 'step_loss': 0, 'trace_loss': 0}
            while not done:
                num_action += 1
                action = self._select_action(state)
                reward, next_state, done = self._step(state, action, num_action)
                next_state_proxy = self._get_state_proxy(next_state)
                
                if 'Commutative' in self.name and self.reward_prediction_type == 'lookup':
                    self.ptr_lst[(tuple(state_proxy), action)] = [reward, next_state_proxy]
                # elif self.reward_prediction_type == 'approximate':
                #     self._add_transition(state_proxy, action, reward, next_state_proxy, prev_state_proxy, prev_action, prev_reward)
                
                losses = self._update_q_table(state_proxy, action, reward, next_state_proxy, done, losses, num_action)

                # prev_state_proxy = state_proxy
                # prev_action = action
                # prev_reward = reward
                
                state = next_state
                state_proxy = next_state_proxy
            
            # if self.reward_prediction_type == 'approximate':
            #     losses = self._update_estimator(losses)
            
            self._decrement_epsilon(episode)
                
            if episode % self.eval_freq == 0:
                eval_return, eval_bridges = self._eval_policy()
                eval_returns.append(eval_return)
                avg_return = np.mean(eval_returns[-self.eval_window:])
                wandb.log({'Average Return': avg_return}, step=episode)
                
                # returns_lst.append(avg_returns)
                
                # if len(returns_lst) > self.eval_window:
                #     min_avg_reward = np.min(returns_lst)
                #     max_avg_reward = np.max(returns_lst)
                #     reward_range = max_avg_reward - min_avg_reward
                    
                #     # reward_diff = ((returns_lst[-1] - returns_lst[-2]) / (reward_range)) * 100                    
                #     # convergence_count = 0 if abs(reward_diff) > self.difference_threshold else convergence_count + 1
                    
                #     reward_var = np.var(returns_lst[-self.eval_window:]) / reward_range
                #     convergence_count = 0 if reward_var > self.variance_threshold else convergence_count + 1
                    
                #     wandb.log({'Reward Variance': reward_var, 'Convergence Count': convergence_count}, step=episode)
                    
                if eval_return > best_return:
                    best_return = eval_return
                    best_bridges = eval_bridges
                    
                # if convergence_count >= self.convergence_window:
                #     terminating_episode = episode
                #     break
    
            traditional_td_errors.append(losses['traditional_td_error'])
            step_losses.append(losses['step_loss'] / (num_action - len(adaptations)))
            trace_losses.append(losses['trace_loss'] / (num_action - len(adaptations)))
            
            avg_traditional_td_errors = np.mean(traditional_td_errors[-self.sma_window:])
            avg_step_loss = np.mean(step_losses[-self.sma_window:])
            avg_trace_loss = np.mean(trace_losses[-self.sma_window:])
            
            wandb.log({
                "Average Traditional TD Error": avg_traditional_td_errors,
                "Average Step Loss": avg_step_loss, 
                "Average Trace Loss": avg_trace_loss}, step=episode)
            
            if losses['commutative_td_error'] != 0:
                commutative_td_errors.append(losses['commutative_td_error'])
                avg_commutative_td_errors = np.mean(commutative_td_errors[-self.sma_window:])
                wandb.log({"Average Commutative TD Error": avg_commutative_td_errors}, step=episode)
            
        return best_return, best_bridges
                        
    def _generate_adaptations(self, problem_instance):
        self.epsilon = self.epsilon_start
        
        self.traditional_update = 0
        self.commutative_update = 0
        
        self.q_table = defaultdict(lambda: {a: 0 for a in range(self.action_dims)})
        # self.estimator = RewardEstimator(self.step_dims, self.estimator_alpha)
        # self.reward_buffer = RewardBuffer(self.buffer_size, self.step_dims, self.action_dims)
                
        self._init_instance(problem_instance)
        self._init_wandb(problem_instance)
        
        best_return, best_bridges = self._train()
            
        wandb.log({'Bridges': best_bridges, 'Return': best_return, 'Traditional Updates': self.traditional_update, 'Commutative Updates': self.commutative_update})
        wandb.finish()
        
        return best_bridges
    
    
class CommutativeQTable(BasicQTable):
    def __init__(self, env, num_instances, random_state, reward_prediction_type):
        super(CommutativeQTable, self).__init__(env, num_instances, random_state, reward_prediction_type)   
        
    """
    Update Rule 0: Traditional Q-Update
    Q(s_1, b) = Q(s_1, b) + alpha * (r_1 + * max_a Q(s', a) - Q(s_1, b))
    Update Rule 1: Commutative Q-Update
    Q(s_2, a) = Q(s_2, a) + alpha * (r_0 - r_2 + r_1 + max_a Q(s', a) - Q(s_2, a))
    """
    def _update_q_table(self, state_proxy, action, reward, next_state_proxy, done, losses, num_action=None):
        losses = super()._update_q_table(state_proxy, action, reward, next_state_proxy, done, losses)
                
        commutative_reward = None
        
        if self.prev_sample is None:
            self.prev_sample = (state_proxy, action, reward)
        elif action != 0:
            prev_state_proxy, prev_action, prev_reward = self.prev_sample
            
            if self.reward_prediction_type == 'lookup':
                commutative_step = self.ptr_lst.get((tuple(prev_state_proxy), action))
                if commutative_step is not None:
                    lookup_reward, commutative_state_proxy = commutative_step
                    commutative_reward = prev_reward + reward - lookup_reward
            # else:            
            #     valid_update = True
            #     commutative_reward = 0
            #     commutative_state_proxy, next_state_proxy = self._reassign_states(prev_state_proxy, action, state_proxy, next_state_proxy)

            if commutative_reward is not None:
                losses = super()._update_q_table(commutative_state_proxy, prev_action, commutative_reward, next_state_proxy, done, losses, traditional_update=False)      
                
            self.prev_sample = (state_proxy, action, reward)
            
        if done:
            self.prev_sample = None
            
        return losses
            
    def _generate_adaptations(self, problem_instance):
        self.ptr_lst = {}
        self.prev_sample = None
        self.commutative_reward_buffer = CommutativeRewardBuffer(self.buffer_size, self.step_dims, self.action_dims)
        
        return super()._generate_adaptations(problem_instance)
    

class GroundedQTable(BasicQTable):
    def __init__(self, env, num_instances, random_state, reward_prediction_type):
        super(GroundedQTable, self).__init__(env, num_instances, random_state, reward_prediction_type)   
        
    def _update_q_table(self, state_proxy, action, reward, next_state_proxy, done, losses, num_action):
        losses = super()._update_q_table(state_proxy, action, reward, next_state_proxy, done, losses)
                
        if self.prev_sample is None:
            self.prev_sample = (state_proxy, action)
        elif action != 0:
            prev_state_proxy, prev_action = self.prev_sample
            
            commutative_state_proxy, next_state_proxy = self._reassign_states(prev_state_proxy, action, state_proxy, next_state_proxy)
            
            commutative_state = self._get_state_from_proxy(commutative_state_proxy)
            next_state = self._get_state_from_proxy(next_state_proxy)
            
            commutative_reward, done = self._get_reward(commutative_state, action, next_state, num_action)
            losses = super()._update_q_table(commutative_state_proxy, prev_action, commutative_reward, next_state_proxy, done, losses, traditional_update=False)
            
        if done:
            self.prev_sample = None
            
        return losses
    
    def _generate_adaptations(self, problem_instance):
        self.prev_sample = None
        
        return super()._generate_adaptations(problem_instance)
    

class RandomQTable(BasicQTable):
    def __init__(self, env, num_instances, random_state, reward_prediction_type):
        super(RandomQTable, self).__init__(env, num_instances, random_state, reward_prediction_type)
        
    def _update_q_table(self, state_proxy, action, reward, next_state_proxy, done, losses, num_action=None):
        losses = super()._update_q_table(state_proxy, action, reward, next_state_proxy, done, losses)
        
        state = self._get_state_from_proxy(state_proxy)
        action = self.rng.integers(self.action_dims)
        next_state = self._get_next_state(state, action)
        reward, done = self._get_reward(state, action, next_state, num_action)
    
        losses = super()._update_q_table(state_proxy, action, reward, next_state_proxy, done, losses, traditional_update=False)
        
        return losses
    
    
class GreedyQTable(BasicQTable):
    def __init__(self, env, num_instances, random_state, reward_prediction_type):
        super(GreedyQTable, self).__init__(env, num_instances, random_state, reward_prediction_type)
        
    def _update_q_table(self, state_proxy, action, reward, next_state_proxy, done, losses, num_action=None):
        losses = super()._update_q_table(state_proxy, action, reward, next_state_proxy, done, losses)
        
        state = self._get_state_from_proxy(state_proxy)
        action = self._select_action(state, is_train=False)
        next_state = self._get_next_state(state, action)
        reward, done = self._get_reward(state, action, next_state, num_action)
        
        losses = super()._update_q_table(state_proxy, action, reward, next_state_proxy, done, losses, traditional_update=False)
        
        return losses
    

class HallucinatedQTable(BasicQTable):
    def __init__(self, env, num_instances, random_state):
        super(HallucinatedQTable, self).__init__(env, num_instances, random_state, None)
                
    def _sample_permutations(self, bridges):
        permutations = {}
        permutations[tuple(bridges)] = None
        
        tmp_action_seq = bridges.copy()
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
    def _hallucinate(self, start_state, bridges, episode_reward, losses):
        permutations = self._sample_permutations(bridges)
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
    
    def _train(self):
        eval_returns = []
        
        for episode in range(self.num_episodes):
            done = False
            bridges = []
            episode_reward = 0
            start_state, adaptations = self._generate_start_state()
            num_action = len(adaptations)
            
            state = start_state
            
            losses = {'traditional_td_error': 0}
            while not done:
                num_action += 1
                action = self._select_action(state)
                reward, next_state, done = self._step(state, action, num_action)  
                             
                state = next_state
                episode_reward += reward
                transformed_action = self._transform_action(action)
                bridges += [transformed_action]
                
            losses = self._hallucinate(start_state, bridges, episode_reward, losses)
            
            self._decrement_epsilon(episode)
            
            if episode % self.eval_freq == 0:
                eval_return, eval_bridges = self._eval_policy()
                eval_returns.append(eval_return)
                avg_return = np.mean(eval_returns[-self.eval_window:])
                # returns_lst.append(avg_returns)
                wandb.log({'Average Return': avg_return}, step=episode)
    
    