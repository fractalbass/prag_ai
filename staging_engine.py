from preprocessor import PreProcessor
import numpy as np
from data_utility import DataUtility


class StagingEngine:

    root_data_dir = "/home/miles/kaggle_tf_audio/data/train/audio"
    class_dirs = ['on', 'off', 'yes', 'no', 'stop', 'go', 'up', 'down', 'left', 'right',
                  'four', 'three', 'bed', 'tree', 'bird', 'happy', 'one', 'two', 'cat', 'house', 'dog',
                  'left', 'seven', 'wow', 'marvin', 'sheila', 'eight', 'nine', 'six', 'zero', 'five']
    target_directory = "/home/miles/kaggle_speech_data"

    preprocessor = None
    du = DataUtility()

    def __init__(self):
        print("Staging engine initializing.")
        self.preprocessor = PreProcessor()
        print("Using preprocessor version: {0}".format(self.preprocessor.__version__))

    def prepare_class(self, root_directory, class_name):
        file_counter = 0
        target_file = "{0}/{1}-{2}".format(self.target_directory, root_directory.split("/")[-1], self.preprocessor.__version__)

        raw_file_names = self.du.get_filenames(root_directory)
        class_features = list()
        # get list of files

        # for each file...
        for filename in raw_file_names:
            file_counter = file_counter + 1
            print(filename)
            full_path = "{0}/{1}".format(root_directory, filename)
            features = self.preprocessor.transform_audio_to_features(full_path)
            class_features.append(features)
        # add the array to a bigger array


        np.savez_compressed(target_file, class_features)
        # compress and save the master array (.npz)
        print("Done preparing {0} files for class [{1}]".format(file_counter, class_name))
        return file_counter

    def run(self):
        print("Staging engine running...")
        total_converted = 0
        for class_dir in self.class_dirs:
            source_dir = "{0}/{1}".format(self.root_data_dir, class_dir)
            total_converted = total_converted + self.prepare_class(source_dir, class_dir)
        print("Total prepared file count: {0}".format(total_converted))


if __name__ == '__main__':
    se = StagingEngine()
    se.run()
