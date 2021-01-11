import unittest


import reward_test

# initialize the test suite
loader = unittest.TestLoader()
suite = unittest.TestSuite()

suite.addTests(loader.loadTestsFromModule(reward_test))

runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)