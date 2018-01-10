import unittest
from preprocessor import PreProcessor


class PreProcessorTest(unittest.TestCase):

    def test_version_number(self):
        p = PreProcessor
        self.assertTrue(p.__version__=='1.0.0')

