import math
import wandb
import random
import itertools
import numpy as np

from agents.utils.ea import EA


class BasicQTable(EA):
    def __init__(self, env, rng):
        super(BasicQTable, self).__init__(env, rng)
                
        self.q_table = None
        # Add a dummy action (+1) to terminate the episode
        self.action_dims = 4 + 1
        self.nS = 2 ** self.state_dims
        
        self.alpha = 0.0005
        self.epsilon_start = 1
        self.num_episodes = 7500
        self.epsilon_decay = 0.999
        self.sma_window = int(self.num_episodes * self.sma_percentage)
        
    def _init_wandb(self, problem_instance):
        config = super()._init_wandb(problem_instance)
        config.alpha = self.alpha
        config.action_cost = self.action_cost
        config.num_episodes = self.num_episodes
        config.epsilon_decay = self.epsilon_decay
        config.percent_obstacles = self.percent_obstacles
        config.action_success_rate = self.action_success_rate
        config.configs_to_consider = self.configs_to_consider

    def _get_state_idx(self, state):
        binary_str = "".join(str(cell) for cell in reversed(state))
        state_idx = int(binary_str, 2)
        return state_idx   

    # Shift action indices to the middle 4 cells of the grid
    # 0 -> 5; 1 -> 6; 2 -> 9; 3 -> 10; 4 -> 4
    def _transform_action(self, action):
        shift = 5
        # Terminating action is unchanged
        if action != 4:
            if action == 2 or action == 3:
                shift += 2
            action += shift
        return action     
        
    def _select_action(self, state):
        if self.rng.random() < self.epsilon:
            original_action = self.rng.integers(self.action_dims)
        else:
            state_idx = self._get_state_idx(state)
            original_action = self.q_table[state_idx].argmax()
            
        transformed_action = self._transform_action(original_action)
        return transformed_action, original_action
    
    def _update_q_table(self, state, action, reward, next_state, done):
        action_idx = action
        state_idx = self._get_state_idx(state)
        next_state_idx = self._get_state_idx(next_state)
        
        td_target = reward + (1 - done) * self.q_table[next_state_idx].max() 
        td_error = td_target - self.q_table[state_idx, action_idx]
        
        self.q_table[state_idx, action_idx] = self.q_table[state_idx, action_idx] + self.alpha * td_error
    
    def _train(self, problem_instance):
        rewards = []
        
        for _ in range(self.num_episodes):
            done = False
            num_action = 0
            action_seq = []
            episode_reward = 0
            state = np.zeros(self.state_dims, dtype=int)
            while not done:
                num_action += 1
                transformed_action, original_action = self._select_action(state)
                reward, next_state, done = self._step(problem_instance, state, transformed_action, num_action)
                
                self._update_q_table(state, original_action, reward, next_state, done)
                state = next_state
                action_seq += [original_action]
                episode_reward += reward
                
            self.epsilon *= self.epsilon_decay
            
            rewards.append(episode_reward)
            avg_rewards = np.mean(rewards[-self.sma_window:])
            wandb.log({"Average Reward": avg_rewards})
                        
    def _get_adaptation(self, problem_instance):
        trials = 25
        scoreboard = set()
        
        for _ in range(trials):
            done = False
            num_action = 0
            action_seq = []
            episode_reward = 0
            state = np.zeros(self.state_dims, dtype=int)
            while not done:
                num_action += 1
                transformed_action, original_action = self._select_action(state)
                reward, next_state, done = self._step(problem_instance, state, transformed_action, num_action)
                state = next_state
                action_seq += [original_action]
                episode_reward += reward
            
            info = tuple(action_seq) + (episode_reward,) 
            scoreboard.add(info)
        
        info = max(scoreboard, key=lambda x: x[-1])
        action_seq = list(info[:-1])
        reward = info[-1]
        
        return action_seq, reward
        
    def _generate_adaptations(self, problem_instance):
        self.epsilon = self.epsilon_start
        self.q_table = np.zeros((self.nS, self.action_dims))
        
        self._init_wandb(problem_instance)
        self._train(problem_instance)
        adaptation, reward = self._get_adaptation(problem_instance)
        
        wandb.log({'Adaptation': adaptation})
        wandb.log({'Reward': reward})
        wandb.finish()
        
        return adaptation


class HallucinatedQTable(BasicQTable):
    def __init__(self, env, rng):
        super(HallucinatedQTable, self).__init__(env, rng)
        self.max_seq_len = 7
                
    def _sample_permutations(self, action_seq):
        permutations = {}
        permutations[tuple(action_seq)] = None
        
        tmp_action_seq = action_seq.copy()
        terminating_action = tmp_action_seq.pop()
        
        if len(tmp_action_seq) <= self.max_seq_len:
            permutations_no_term = itertools.permutations(tmp_action_seq)
            for permutation in permutations_no_term:
                permutations[tuple(permutation) + (terminating_action,)] = None
        else:
            i = 1
            max_samples = math.factorial(self.max_seq_len)
            while i < max_samples:
                random.shuffle(tmp_action_seq)
                permutations[tuple(tmp_action_seq) + (terminating_action,)] = None
                i += 1   
        
        return list(permutations.keys())       
    
    # Hallucinate episodes by permuting the action sequence to simulate commutativity
    def _hallucinate(self, action_seq, episode_reward):
        seq_len = len(action_seq)
        reward = episode_reward if seq_len == 1 else episode_reward / (seq_len - 1)

        permutations = self._sample_permutations(action_seq)
        for permutation in permutations:
            state = np.zeros(self.state_dims, dtype=int)
            for action in permutation:
                next_state = state.copy()
                
                done = action == action_seq[-1]
                transformed_action = self._transform_action(action)
                
                if done and seq_len != 1:
                    hallucinated_reward = 0
                elif done and seq_len == 1:
                    hallucinated_reward = reward
                else:
                    hallucinated_reward = reward
                    next_state[transformed_action] = 1
                    
                self._update_q_table(state, action, hallucinated_reward, next_state, done)
                state = next_state
    
    def _train(self, problem_instance):
        rewards = []
        
        for _ in range(self.num_episodes):
            done = False
            num_action = 0
            action_seq = []
            episode_reward = 0
            state = np.zeros(self.state_dims, dtype=int)
            while not done:
                num_action += 1
                transformed_action, original_action = self._select_action(state)
                reward, next_state, done = self._step(problem_instance, state, transformed_action, num_action)   
                             
                state = next_state
                action_seq += [original_action]
                episode_reward += reward
                
            self._hallucinate(action_seq, episode_reward)
            self.epsilon *= self.epsilon_decay
            
            rewards.append(episode_reward)
            avg_rewards = np.mean(rewards[-self.sma_window:])
            wandb.log({"Average Reward": avg_rewards})
    
    
class CommutativeQTable(BasicQTable):
    def __init__(self, env, rng):
        super(CommutativeQTable, self).__init__(env, rng)
        
        # (s, a) -> (s', r)
        self.ptr_lst = {}
        self.previous_sample = None
    
    """
    Update Rule 0: Traditional Q-Update
    Q(s_1, b) = Q(s_1, b) + alpha * (r_1 + * max_a Q(s', a) - Q(s_1, b))
    Update Rule 1: Commutative Q-Update
    Q(s_2, a) = Q(s_2, a) + alpha * (r_0 - r_2 + r_1 + max_a Q(s', a) - Q(s_2, a))
    """
    def _update_q_table(self, state, action, reward, next_state, done):
        super()._update_q_table(state, action, reward, next_state, done)
        
        if self.previous_sample is None:
            s = self._get_state_idx(state)
            a = action
            r_0 = reward
            s_1 = self._get_state_idx(next_state)
            
            self.previous_sample = (s, a, r_0)
            self.ptr_lst[(s, a)] = (s_1, r_0)
        else:
            s, a, r_0 = self.previous_sample
            s_1 = self._get_state_idx(state)
            b = action
            r_1 = reward
            s_prime = self._get_state_idx(next_state)
            
            if (s, b) in self.ptr_lst:
                s_2, r_2 = self.ptr_lst[(s, b)]
                
                td_target = r_0 - r_2 + r_1 + (1 - done) * self.q_table[s_prime].max()
                td_error = td_target - self.q_table[s_2, a]
                
                self.q_table[s_2, a] = self.q_table[s_2, a] + self.alpha * td_error
                
            self.previous_sample = (s_1, b, r_1)
            self.ptr_lst[(s_1, b)] = (s_prime, r_1)
            
        if done:
            self.previous_sample = None