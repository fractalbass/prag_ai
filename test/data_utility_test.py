import unittest
from data_utility import DataUtility

class PreprocessorTest(unittest.TestCase):

    def test_one_plus_one(self):
        self.assertEqual(1+1, 2)

    def test_get_filenames(self):
        du = DataUtility()
        files = du.get_filenames('.')
        self.assertTrue('preprocessor_test.py' in files)
