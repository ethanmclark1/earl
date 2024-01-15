import math
import wandb
import random
import itertools
import numpy as np

from agents.utils.ea import EA
from scipy.optimize import minimize

class BasicQTable(EA):
    def __init__(self, env, rng, random_state, reward_prediction_type=None):
        super(BasicQTable, self).__init__(env, rng, random_state)
                
        self.q_table = None
        # Add a dummy action (+1) to terminate the episode
        self.nS = 2 ** 16

        self.alpha = 0.0004
        self.max_seq_len = 7
        self.epsilon_start = 1
        self.sma_window = 2500
        self.min_epsilon = 0.05
        self.reward_window = 100
        self.num_episodes = 250000
        self.reward_prediction_type = reward_prediction_type
        self.epsilon_decay = 0.995 if random_state else 0.999975
        
    def _init_wandb(self, problem_instance):
        config = super()._init_wandb(problem_instance)
        config.alpha = self.alpha
        config.sma_window = self.sma_window
        config.max_action = self.max_action
        config.action_cost = self.action_cost
        config.min_epsilon = self.min_epsilon
        config.max_seq_len = self.max_seq_len
        config.random_state = self.random_state
        config.num_episodes = self.num_episodes
        config.reward_window = self.reward_window
        config.epsilon_decay = self.epsilon_decay
        config.percent_holes = self.percent_holes
        config.action_success_rate = self.action_success_rate
        config.configs_to_consider = self.configs_to_consider
        config.reward_prediction_type = self.reward_prediction_type
        
    def _decrement_epsilon(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.min_epsilon)
        
    def _select_action(self, state):
        if self.rng.random() < self.epsilon:
            original_action = self.rng.integers(self.action_dims)
        else:
            state_idx = self._get_state_idx(state)
            original_action = self.q_table[state_idx].argmax()
            
        transformed_action = self._transform_action(original_action)
        return transformed_action, original_action
    
    def _update_q_table(self, state, action, reward, next_state, done, episode):
        s = self._get_state_idx(state)
        a = action
        s_prime = self._get_state_idx(next_state)
        
        td_target = reward + (1 - done) * self.q_table[s_prime].max() 
        td_error = td_target - self.q_table[s, a]
        
        self.q_table[s, a] += self.alpha * td_error
    
    def _train(self, problem_instance):
        rewards = []
        
        best_reward = -np.inf
        best_action_seq = None
        
        for episode in range(self.num_episodes):
            done = False
            action_seq = []
            episode_reward = 0
            state, bridges = self._get_state(problem_instance)
            num_action = len(bridges)
            while not done:
                num_action += 1
                transformed_action, original_action = self._select_action(state)
                reward, next_state, done = self._step(problem_instance, state, transformed_action, num_action)
                
                self._update_q_table(state, original_action, reward, next_state, done, episode)
                state = next_state
                episode_reward += reward
                action_seq += [original_action]
                
            self._decrement_epsilon()
            
            rewards.append(episode_reward)
            avg_rewards = np.mean(rewards[-self.sma_window:])
            wandb.log({"Average Reward": avg_rewards}, step=episode)
            
            if episode_reward > best_reward:
                best_action_seq = action_seq
                best_reward = episode_reward
            
        return best_action_seq, best_reward
    
    def _get_final_adaptation(self, problem_instance):
        best_reward = -np.inf
        best_action_seq = None
        
        for _ in range(25):
            done = False
            num_action = 0
            action_seq = []
            self.epsilon = 0
            episode_reward = 0
            state, _ = self._generate_fixed_state(problem_instance)
            while not done:
                num_action += 1
                transformed_action, original_action = self._select_action(state)
                reward, next_state, done = self._step(problem_instance, state, transformed_action, num_action)
                
                state = next_state
                action_seq += [original_action]
                episode_reward += reward
                
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_action_seq = action_seq
            
            return best_action_seq, best_reward
        
    def _generate_adaptations(self, problem_instance):
        self.epsilon = self.epsilon_start
        self.q_table = np.zeros((self.nS, self.action_dims))
        
        self._set_max_action(problem_instance)
        self._init_wandb(problem_instance)
        best_adaptation, best_reward = self._train(problem_instance)
        
        if self.random_state:
            best_adaptation, best_reward = self._get_final_adaptation(problem_instance)
        
        wandb.log({'Adaptation': best_adaptation, 'Final Reward': best_reward})
        wandb.finish()
        
        return best_adaptation
    
    
class CommutativeQTable(BasicQTable):
    def __init__(self, env, rng, random_state, reward_prediction_type):
        super(CommutativeQTable, self).__init__(env, rng, random_state, reward_prediction_type)
    
    # Predict r_3 based on R(s,a,s')
    def _predict_reward(self, s, b, s_2, a, s_prime, r_0, r_1):
        def objective(w, X_0, X_1, y):
            r2_pred = np.dot(X_0, w[:-1]) + w[-1]
            r3_pred = np.dot(X_1, w[:-1]) + w[-1]
            y_pred = r2_pred + r3_pred
            return np.mean(np.square(y - y_pred))

        loss = 0
        s_idx = self._get_state_idx(s)
        s_2_idx = self._get_state_idx(s_2)
        s_prime_idx = self._get_state_idx(s_prime)
        
        self.feature_vector_r2.append([s_idx, b, s_2_idx])
        self.feature_vector_r3.append([s_2_idx, a, s_prime_idx])
        self.reward_y.append([r_0 + r_1])
        
        # Start from one to account for first sample
        if len(self.feature_vector_r2) % self.reward_window == 1:
            feature_vector_r2 = np.array(self.feature_vector_r2)
            feature_vector_r3 = np.array(self.feature_vector_r3)
            reward_y = np.array(self.reward_y)
            init_guess = np.zeros(len(feature_vector_r2[0]) + 1)
            res = minimize(objective, init_guess, args=(feature_vector_r2, feature_vector_r3, reward_y), method='SLSQP')
            self.w = res.x
            loss = res.fun

        feature_vector = np.array([s_2_idx, a, s_prime_idx])
        r3_pred = np.dot(feature_vector, self.w[:-1]) + self.w[-1]
        
        return r3_pred, loss
    
    # Predict r_2 based on r_0 and r_1, then calculate r_3 from r_0 + r_1 - r_2
    def _predict_noisy_reward(self, r_0, r_1, r_2):
        def objective(w, X, y):
            r2_pred = np.dot(X, w[:-1]) + w[-1]
            return np.mean(np.square(y - r2_pred))
        
        loss = 0
        feature_vector = np.array([r_0, r_1])
        self.noisy_reward_X.append(feature_vector)
        self.noisy_reward_y.append(r_2)
        
        if len(self.noisy_reward_X) % self.reward_window == 0:
            if len(self.noisy_reward_X) > self.reward_window:
                noisy_reward_X = np.array(self.noisy_reward_X)
                noisy_reward_y = np.array(self.noisy_reward_y)
                init_guess = np.zeros(len(noisy_reward_X[0]) + 1)
                res = minimize(objective, init_guess, args=(noisy_reward_X, noisy_reward_y), method='SLSQP')
                self.noisy_w = res.x
                loss = res.fun
            
        r2_pred = np.dot(feature_vector, self.noisy_w[:-1]) + self.noisy_w[-1] if self.noisy_w is not None else r_2
        r_3 = r_0 + r_1 - r2_pred
        
        return r_3, loss

    # Predict r_3 based on r_0, r_1, r_2
    def _predict_noisiest_reward(self, r_0, r_1, r_2):
        def objective(w, X, y):
            r2 = X[:, 2]
            r3_pred = np.dot(X, w[:-1]) + w[-1] 
            return np.mean(np.square(y - (r2 + r3_pred)))
        
        loss = 0
        feature_vector = np.array([r_0, r_1, r_2])
        self.noisest_reward_X.append(feature_vector)
        self.noisest_reward_y.append(r_0 + r_1)
        
        if len(self.noisest_reward_X) % self.reward_window == 0:
            if len(self.noisest_reward_X) > self.reward_window:
                noisest_reward_X = np.array(self.noisest_reward_X)
                noisest_reward_y = np.array(self.noisest_reward_y)
                init_guess = np.zeros(len(noisest_reward_X[0]) + 1)
                res = minimize(objective, init_guess, args=(noisest_reward_X, noisest_reward_y), method='SLSQP')
                self.noisest_w = res.x
                loss = res.fun
        
        r3_pred = np.dot(feature_vector, self.noisest_w[:-1]) + self.noisest_w[-1] if self.noisest_w is not None else r_0 + r_1 - r_2
        
        return r3_pred, loss
        
    """
    Update Rule 0: Traditional Q-Update
    Q(s_1, b) = Q(s_1, b) + alpha * (r_1 + * max_a Q(s', a) - Q(s_1, b))
    Update Rule 1: Commutative Q-Update
    Q(s_2, a) = Q(s_2, a) + alpha * (r_0 - r_2 + r_1 + max_a Q(s', a) - Q(s_2, a))
    """
    def _update_q_table(self, state, action, reward, next_state, done, episode):
        super()._update_q_table(state, action, reward, next_state, done, episode)
        
        # If its linear then don't store transition table
        if self.reward_prediction_type != 'linear':
            state_idx = self._get_state_idx(state)
            if 'mean' in self.reward_prediction_type or 'median' in self.reward_prediction_type:
                if (state_idx, action) in self.ptr_lst:
                    self.ptr_lst[(state_idx, action)][0] += [reward]
                else:
                    reward_lst = [reward]
                    self.ptr_lst[(state_idx, action)] = [reward_lst, next_state]
            else:
                self.ptr_lst[(state_idx, action)] = [reward, next_state]
        
        if self.previous_sample is None:
            self.previous_sample = (state, action, reward)
        else:
            s, a, r_0 = self.previous_sample
            s_1 = state
            b = action
            r_1 = reward
            s_prime = next_state
            
            s_idx = self._get_state_idx(s)
            if (s_idx, b) in self.ptr_lst:
                r_2_lst, s_2 = self.ptr_lst[(s_idx, b)]      
                if  'mean' in self.reward_prediction_type:
                    r_2 = np.mean(r_2_lst)
                elif 'median' in self.reward_prediction_type:
                    r_2 = np.median(r_2_lst)
                else:
                    r_2 = r_2_lst

                if 'table' in self.reward_prediction_type:
                    r_3 = r_0 + r_1 - r_2
                elif 'noisy' in self.reward_prediction_type:
                    r_3, estimation_loss = self._predict_noisy_reward(r_0, r_1, r_2)
                    if estimation_loss != 0:
                        wandb.log({"Estimation Loss": estimation_loss}, step=episode)
                elif 'noisiest' in self.reward_prediction_type:
                    r_3, estimation_loss = self._predict_noisiest_reward(r_0, r_1, r_2)
                    if estimation_loss != 0:
                        wandb.log({"Estimation Loss": estimation_loss}, step=episode)
                
                super()._update_q_table(s_2, a, r_3, s_prime, done, episode)
                        
            elif self.reward_prediction_type == 'linear':
                transformed_action = self._transform_action(b)
                s_2 = self._get_next_state(s, transformed_action)
                
                r_3, estimation_loss = self._predict_reward(s, b, s_2, a, s_prime, r_0, r_1)
                if estimation_loss != 0:
                    wandb.log({"Estimation Loss": estimation_loss}, step=episode)

                super()._update_q_table(s_2, a, r_3, s_prime, done, episode)
                
            self.previous_sample = (s_1, b, r_1)
            
        if done:
            self.previous_sample = None
            
    def _generate_adaptations(self, problem_instance):
        self.previous_sample = None
        
        # Table
        self.ptr_lst = {}
        # Linear
        self.w = None
        self.feature_vector_r2 = []
        self.feature_vector_r3 = []
        self.reward_y = []
        # Noisy Linear
        self.noisy_w = None
        self.noisy_reward_X = []
        self.noisy_reward_y = []
        # Noisest Linear
        self.noisest_w = None
        self.noisest_reward_X = []
        self.noisest_reward_y = []
        
        return super()._generate_adaptations(problem_instance)


class HallucinatedQTable(BasicQTable):
    def __init__(self, env, rng, random_state):
        super(HallucinatedQTable, self).__init__(env, rng, random_state)
                
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
            start_state, bridges = self._get_state(problem_instance)
            num_action = len(bridges)
            state = start_state
            while not done:
                num_action += 1
                transformed_action, original_action = self._select_action(state)
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