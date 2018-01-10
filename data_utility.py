from tensorflow.python.lib.io import file_io
from keras import models
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import os
import numpy as np
from time import time
import itertools
class DataUtility:

    #training_categories = ['on', 'off', 'yes', 'no', 'stop', 'go', 'up', 'down', 'left', 'right']


    def get_filenames(self, root_folder):
        return os.listdir(root_folder)

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

    def save_model(self, prefix, model):

        ts = str(time())
        filename = '{0}_{1}_{2}.h5'.format(prefix, "model", ts)
        model.save(filename)

        # Save the model to the Cloud Storage bucket's jobs directory
        try:
            with file_io.FileIO(filename, mode='r') as input_f:
                gs_name = "gs://{0}/{1}".format(self.bucket_id, filename)
                with file_io.FileIO(gs_name, mode='w+') as output_f:
                    output_f.write(input_f.read())
        except Exception as er:
            print("An error occurred.  {0}".format(er.message))

    def save_multi_model(self, path, model_name, model):

        ts = str(time())
        filename = '{0}/{1}.h5'.format(path, model_name)
        model.save(filename)

        # Save the model to the Cloud Storage bucket's jobs directory
        # try:
        #     with file_io.FileIO(filename, mode='r') as input_f:
        #         gs_name = "gs://{0}/models/{1}".format(self.bucket_id, filename)
        #         with file_io.FileIO(gs_name, mode='w+') as output_f:
        #             output_f.write(input_f.read())
        # except Exception as er:
        #   er  print("An error occurred.  {0}".format(er.message))


    def save_categories(self, filename):
        with file_io.FileIO("{0}.p".format(filename), mode='w+') as output_f:
            output_f.write(self.training_categories)