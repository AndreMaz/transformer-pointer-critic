import numpy as np
import random
from environment.base.base import BaseEnvironment
from environment.custom.sorting_numbers.sequence import Sequence


class SortingEnvironment(BaseEnvironment):
    def __init__(self, name: str, opts: dict):
        super(SortingEnvironment, self).__init__(name)

        self.normalize = opts['normalize']
        self.add_special_chars = opts['add_special_chars']
        
        self.min_value = opts['min_value']
        self.max_value = opts['max_value']

        self.seq_length = opts['seq_length']
        self.use_simple_reward = opts['use_simple_reward']
        self.generate_new_after_reset = opts['generate_new_after_reset']

        self.sequence = Sequence(
            self.min_value,
            self.max_value,
            self.seq_length,
            self.normalize,
            self.add_special_chars)

        # Store the env, the solution and one-hot encoded solution
        self.env, self.env_solution, self.env_correct_actions = self.sequence.generate_sequence()
        # Create a copy for easy reseting the env
        self.env_backup = self.env.copy()

        self.action_space = len(self.env)
        self.observation_space = len(self.env)

        self.step_count = 0

        self.selected_actions = []

    def reset(self):
        self.step_count = 0
        self.selected_actions = []

        if not self.generate_new_after_reset:
            self.env = self.env_backup.copy()
        else:
            # Store the env, the solution and one-hot encoded solution
            self.env, self.env_solution, self.env_correct_actions = self.sequence.generate_sequence()
            # Create a copy for easy reseting the env
            self.env_backup = self.env.copy()

        return self.env.copy()

    def state(self):
        return self.env.copy()

    def step(self, action):
        self.selected_actions.append(int(action))

        # Number selected: mask it out
        self.env[action] = self.sequence.MASK

        if self.use_simple_reward:
            reward = self.simple_reward(action)
        else:
            reward = self.complex_reward(action)
        
        # Update step count
        self.step_count += 1

        isDone = False
        if (self.step_count == self.observation_space):
            isDone = True

        info = {
            'seq_size': self.observation_space,
            "num_masked_elements": len(np.where(self.env == self.sequence.MASK)[0])}

        return self.env.copy(), reward, isDone, info
    
    def multiple_steps(self, actions):
        rewards = []

        for action in actions:
            next_step, reward, isDone, info = self.step(action)
            # Append the reward
            rewards.append(reward)
        
        return next_step, rewards, isDone, info

    def simple_reward(self, action):
        if (self.step_count >= self.observation_space):
            raise IndexError('All actions has been taken already')

        # Default value
        reward = -1.0
        # Correct action. Update reward
        if (self.env_correct_actions[self.step_count][action] == 1):
            reward = 1.0
        
        return reward

    def complex_reward(self, action):
        if (self.step_count >= self.observation_space):
            raise IndexError('All actions has been taken already')

        # Correct action. Update reward
        if (self.env_correct_actions[self.step_count][action] == 1):
            reward = 1.0
        else:
            # Find the index of correct action
            correct_action = np.where(self.env_correct_actions[self.step_count] == 1)[0][0]
            # Compute the difference
            reward = - abs(correct_action - action)

        return reward

    def sample_action(self):
        return random.randint(0, len(self.env) - 1)

    def get_action_space(self):
        return self.action_space
    
    def get_observation_space(self):
        return self.observation_space
    
    def seed(self, seed):
        return

    def close(self):
        return
    
    def add_stats_to_agent_config(self, agent_config):
        agent_config['action_space'] = self.get_action_space()
        agent_config['observation_space'] = self.get_observation_space()
        agent_config['vocab_size'] = self.sequence.vocab_size
        agent_config['SOS_CODE'] = self.sequence.SOS

        return agent_config

    def print_stats(self):
        print('------------------------------------------------------')
        print('Numbers to sort')
        if self.normalize:
            print(self.sequence.denormalize_num_sequence(self.env_backup))
        else: 
            print(self.env_backup)

        print('Correct Solution')
        if self.normalize:
            print(self.sequence.denormalize_num_sequence(self.env_solution))
        else:
            print(self.env_solution)

        agent_solution = []
        # self.selected_actions.append(3)
        for action in self.selected_actions:
            agent_solution.append(self.env_backup[action])

        print('Actual Solution')
        if self.normalize:
            print(self.sequence.denormalize_num_sequence(agent_solution))
        else:
            print(agent_solution) 

        print('Correct action selection sequence')
        correct_actions = []
        for i in range(self.action_space):
            index = np.where(self.env_correct_actions[i] == 1)[0][0]
            correct_actions.append(int(index))

        print(correct_actions)

        print('Actual action selection sequence')
        print(self.selected_actions)
        print('------------------------------------------------------')
