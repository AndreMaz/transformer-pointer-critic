import tensorflow as tf
import random

class BaseAgent():
    def __init__(self, name: str, opts: dict):
        self.agent_name = name

        # Store remaining options
        self.opts = opts

    def name(self):
        return self.agent_name.upper()

    def get_config(self):
        return self.opts

    def build_model(self):
        raise NotImplementedError('Method "buildModel" not implemented')

    def print_model_summary(self):
        raise NotImplementedError('Method "getModelSummary" not implemented')

    def act(self, state):
        raise NotImplementedError('Method "act" not implemented')

    def replay(self, batch_size):
        raise NotImplementedError('Method "replay" not implemented')
