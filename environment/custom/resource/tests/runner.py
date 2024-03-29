import unittest

import penalty_test
import reward_test
import env_test
import utils_test

# initialize the test suite
loader = unittest.TestLoader()
suite = unittest.TestSuite()

suite.addTests(loader.loadTestsFromModule(penalty_test))
suite.addTests(loader.loadTestsFromModule(reward_test))
suite.addTests(loader.loadTestsFromModule(env_test))
suite.addTests(loader.loadTestsFromModule(utils_test))

runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)