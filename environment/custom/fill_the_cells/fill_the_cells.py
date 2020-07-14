# Link https://github.com/shakedzy/notebooks/tree/master/q_learning_and_dqn
import numpy as np
import random
from environment.base.base import BaseEnvironment


class FillTheCells(BaseEnvironment):
    def __init__(self, name: str, opts: dict):
        super(FillTheCells, self).__init__(name)

        self.board_size = 10
        self.reset()

    def reset(self):
        self.env = np.zeros(self.board_size)
        return self.env.copy()

    def state(self):
        return self.env.copy()

    def step(self, cell):
        if (cell > self.board_size - 1): raise ValueError(f"Cell index '{cell}' is out of bound!")
        # returns: next_state, reward, game_over?, info

        # By default the game isn't finished
        isDone = False

        if self.env[cell] == 0:
            # Update cell
            self.env[cell] = 1
            isDone = (self.env == 1).all()
            reward = 1.0
        else:
            # End after wrong decision? If not set to False
            isDone = True
            reward = -1.0

        info = {"board_size": self.board_size,
                "num_empty_cells": len(np.where(self.env == 0)[0])}

        return self.env.copy(), reward, isDone, info

    def sample_action(self):
        return random.randint(0, len(self.board_size) - 1)

    def get_action_space(self):
        return self.board_size

    def close(self):
        return

    def seed(self, seed):
        return

    def get_observation_space(self):
        return self.board_size

    def add_stats_to_agent_config(self, agent_config):
        agent_config['action_space'] = self.get_action_space()
        agent_config['observation_space'] = self.get_observation_space()
        
        return agent_config
    
    def print_stats(self):
        print('---------Environment Stats-----------------')
        print(f'Name: "{self.name}"')
        print('___________________________________________')

        print('Action Space')
        print(self.board_size)

        print('Observation Space')
        print(self.board_size)
        print('-------------------------------------------')