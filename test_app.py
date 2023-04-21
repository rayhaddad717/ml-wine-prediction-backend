import unittest

# Discover and run all tests in the 'tests' folder
if __name__ == "__main__":
    test_suite = unittest.TestLoader().discover('tests')
    unittest.TextTestRunner().run(test_suite)
