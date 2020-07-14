import unittest

import backpack_test
import item_test
import env_test

# initialize the test suite
loader = unittest.TestLoader()
suite = unittest.TestSuite()

suite.addTests(loader.loadTestsFromModule(backpack_test))
suite.addTests(loader.loadTestsFromModule(item_test))
suite.addTests(loader.loadTestsFromModule(env_test))

runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)