import gym
from environment.base.base import BaseEnvironment


class GymEnvironment(BaseEnvironment):
    def __init__(self, name: str):
        super(GymEnvironment, self).__init__(name)

        self.env = gym.make(name)
        self.env.reset()

    def reset(self):
        # Reset the env and return initial observation
        return self.env.reset()

    def state(self):
        # Return current state
        return self.env.state

    def step(self, action):
        return self.env.step(action)

    def sample_action(self):
        # Return random action
        return self.env.action_space.sample()

    def get_action_space(self):
        # Returns action space
        if hasattr(self.env.action_space, 'n'):
            return self.env.action_space.n
        else:
            return self.env.action_space.shape[0]

    def seed(self, num):
        self.env.seed(num)

    def close(self):
        self.env.close()

    def get_observation_space(self):
        # Returns observation space
        return self.env.observation_space.shape[0]

    def add_stats_to_agent_config(self, agent_config):
        agent_config['action_space'] = self.get_action_space()
        agent_config['observation_space'] = self.get_observation_space()
        
        return agent_config

    def print_stats(self):
        print('---------Environment Stats-----------------')
        print(f'Name: "{self.name}"')
        print('___________________________________________')
        print('Action Space')
        print(self.env.action_space)

        print('Observation Space')
        print(self.env.observation_space)

        print('Observation Space Higher Bound')
        print(self.env.observation_space.high)

        print('Observation Space Lower Bound')
        print(self.env.observation_space.low)

        print('Maximum Episode Steps')
        print(self.env.spec.max_episode_steps)
        print('-------------------------------------------')
