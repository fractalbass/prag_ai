import unittest
from data_utility import DataUtility
import numpy as np
import os

class PreprocessorTest(unittest.TestCase):

    def test_one_plus_one(self):
        self.assertEqual(1+1, 2)

    def test_get_filenames(self):
        du = DataUtility()
        files = du.get_filenames('.')
        self.assertTrue('preprocessor_test.py' in files)

    def test_load_npz_training_data(self):
        du = DataUtility()
        x,y,preprocessor_version = du.load_npz_training_data("./test_npz_data")
        self.assertTrue(len(x) == len(y))
        self.assertTrue(len(x) == 9486)
        self.assertTrue(preprocessor_version == '1.0.0')

    def test_get_category_name(self):
        du = DataUtility()
        self.assertTrue(du.get_category_name_from_filename('up-1.0.0.npz') == 'up')


    def test_get_preprocessor_version_from_npz(self):
        du = DataUtility()
        self.assertTrue(du.get_preprocessor_version_from_filename('up-1.0.0.npz') == '1.0.0')

    def test_load_npz_training_data_preprocessor_version_mismatch(self):
        du = DataUtility()
        with self.assertRaises(Exception):
            x, y, preprocessor_version = du.load_npz_training_data("./test_npz_mismatch")

    def test_get_vectorized_data(self):
        the_data = ['house', 'cat', 'boo', 'house', 'up', 'no', 'up']
        du = DataUtility()
        classes, vector = du.get_vectorized_data(the_data)
        self.assertTrue(classes == ['house', 'cat', 'boo', 'up', 'no'])
        self.assertTrue(np.array_equal(vector[0], [1, 0, 0, 0, 0]))
        self.assertTrue(np.array_equal(vector[1], [0, 1, 0, 0, 0]))
        self.assertTrue(np.array_equal(vector[2], [0, 0, 1, 0, 0]))
        self.assertTrue(np.array_equal(vector[3], [1, 0, 0, 0, 0]))
        self.assertTrue(np.array_equal(vector[4], [0, 0, 0, 1, 0]))
        self.assertTrue(np.array_equal(vector[5], [0, 0, 0, 0, 1]))
        self.assertTrue(np.array_equal(vector[6], [0, 0, 0, 1, 0]))
        self.assertTrue(len(the_data), len(vector))

    def test_save_load_file(self):
        dir = '.'
        du = DataUtility()
        files = os.listdir(".")
        for file in files:
            if file.endswith(".p"):
                os.remove(os.path.join(dir,file))

        categories = ['one','two','three']
        saved_file_name = du.save_categories('.', '1.0.0', categories)
        loaded_categories = du.load_categories(saved_file_name)
        self.assertTrue(np.array_equal(categories, loaded_categories))


if __name__ == '__main__':
    unittest.main()