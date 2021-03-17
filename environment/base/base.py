
class BaseEnvironment(): # pragma: no cover
    def __init__(self, name: str):
        self.name = name

    def reset(self):
        raise NotImplementedError('Method "reset" not implemented')

    def state(self):
        raise NotImplementedError('Method "state" not implemented')

    def step(self, action):
        raise NotImplementedError('Method "step" not implemented')

    def sample_action(self):
        raise NotImplementedError('Method "sample_action" not implemented')

    def get_action_space(self):
        raise NotImplementedError('Method "get_action_space" not implemented')

    def seed(self, num):
        raise NotImplementedError('Method "seed" not implemented')

    def close(self):
        raise NotImplementedError('Method "close" not implemented')

    def get_observation_space(self):
        raise NotImplementedError(
            'Method "get_observation_space" not implemented')

    def print_stats(self):
        raise NotImplementedError('Method "print_stats" not implemented')
