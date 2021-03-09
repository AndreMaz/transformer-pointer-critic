import sys
sys.path.append('./tests/unit')
import unittest


import agent.agent_test

import environment.resource_v3.env_test
import environment.resource_v3.heuristic_test
import environment.resource_v3.node_test
import environment.resource_v3.reward_test
import environment.resource_v3.utils_test


# initialize the test suite
loader = unittest.TestLoader()
suite = unittest.TestSuite()

# Agent Unit tests
suite.addTests(loader.loadTestsFromModule(agent.agent_test))
# Resource V3 unit tests
suite.addTests(loader.loadTestsFromModule(environment.resource_v3.env_test))
suite.addTests(loader.loadTestsFromModule(environment.resource_v3.heuristic_test))
suite.addTests(loader.loadTestsFromModule(environment.resource_v3.node_test))
suite.addTests(loader.loadTestsFromModule(environment.resource_v3.reward_test))
suite.addTests(loader.loadTestsFromModule(environment.resource_v3.utils_test))

runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)