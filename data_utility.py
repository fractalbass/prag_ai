from tensorflow.python.lib.io import file_io
from keras import models
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import os
import numpy as np
from time import time
import pickle
import itertools
from keras import utils
from collections import OrderedDict

class DataUtility:


    #training_categories = ['on', 'off', 'yes', 'no', 'stop', 'go', 'up', 'down', 'left', 'right']


    def get_filenames(self, root_folder):
        return os.listdir(root_folder)

    def load_npz_training_data(self, root_folder):
        x = list()
        y = list()
        preprocessor_version = None
        for d in os.listdir(root_folder):
            if d.split('.')[-1] == 'npz':
                ppv = self.get_preprocessor_version_from_filename(d)

                if preprocessor_version is None:
                    preprocessor_version = ppv
                elif ppv != preprocessor_version:
                    raise Exception("Preprocessor version mismatch in training data folder {0}!  ".format(root_folder))

                full_path = "{0}/{1}".format(root_folder, d)
                category_name = self.get_category_name_from_filename(d)
                class_data = np.load(full_path)
                for feature_set in class_data.items():
                    for f in feature_set[1]:
                        x.append(f)
                        y.append(category_name)
                print("Class {0} loaded.".format(category_name))
            else:
                print("Unknown directory {0}.  Skipping.".format(d))
        print("Data load complete.")
        return x, y, preprocessor_version

    def load_data_local(self, root_folder, training_categories, other_categories):
        x = list()
        y = list()
        for d in os.listdir(root_folder):
            if d in training_categories:
                full_path = "{0}/{1}".format(root_folder, d)
                category_name = d.split(".")[0]
                class_data = np.load(full_path)
                for feature_set in class_data.items():
                    for f in feature_set[1]:
                        x.append(f)
                        y.append(training_categories.index(category_name))
                print("Class {0} loaded.".format(category_name))
            elif d in other_categories:
                full_path = "{0}/{1}".format(root_folder, d)
                category_name = "other"
                class_data = np.load(full_path)
                for feature_set in class_data.items():
                    for f in feature_set[1]:
                        x.append(f)
                        y.append(len(training_categories))  #  The last category will be other.
            else:
                print("Unknown directory {0}.  Skipping.".format(d))
        print("Data load complete.")
        return x, y

    def load_local_binary_data(self, root_folder, target):
        x_matches = list()
        y_matches = list()

        x_other = list()
        y_other = list()
        for d in os.listdir(root_folder):
            if d.split('.')[0] == target:
                full_path = "{0}/{1}".format(root_folder, d)
                class_data = np.load(full_path)
                for feature_set in class_data.items():
                    for f in feature_set[1]:
                        x_matches.append(f)
                        y_matches.append(0)
                print("Class {0} loaded.".format(d))
            elif ".npz" in d:
                full_path = "{0}/{1}".format(root_folder, d)
                class_data = np.load(full_path)
                for feature_set in class_data.items():
                    for f in feature_set[1]:
                        x_other.append(f)
                        y_other.append(1)  #  The last category will be other.
            else:
                print("Unknown directory {0}.  Skipping.".format(d))


        print("Data load complete.")
        stp = int(len(x_other)/len(x_matches))
        reduced_other_x = [x_other[x] for x in range(0, len(x_other), stp)]
        reduced_other_y = [y_other[x] for x in range(0, len(y_other), stp)]

        for x in reduced_other_x:
            x_matches.append(x)

        for y in reduced_other_y:
            y_matches.append(y)

        return x_matches, y_matches

    def save_model(self, saved_model_dir, prefix, model):

        ts = str(time())
        filename = '{0}/{1}_{2}_{3}.h5'.format(saved_model_dir, prefix, "model", ts)
        model.save(filename)

    def save_categories(self, saved_model_dir, prefix, training_categories):
        ts = str(time())
        filename = '{0}/{1}_{2}_{3}.p'.format(saved_model_dir, prefix, "categories", ts)
        with open(filename, mode="wb") as f:
            pickle.dump(training_categories, f)
        return filename

    def load_categories(self, filename):
        with open(filename, 'rb') as f:
            loaded_list = pickle.load(f)
        return loaded_list

    def get_category_name_from_filename(self, filename):
        #blah-1.2.3.npz
        return filename.split('-')[0]

    def get_preprocessor_version_from_filename(self, filename):
        #blah-1.2.3.npz
        return filename.split('-')[1][:-4]

    def get_vectorized_data(self, a):
        classes = list(OrderedDict.fromkeys(a))
        idx = [classes.index(i) for i in list(a)]
        vectors = utils.to_categorical(idx, len(classes))
        return classes, vectors
