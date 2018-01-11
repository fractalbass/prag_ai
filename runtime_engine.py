from data_utility import DataUtility
from preprocessor import PreProcessor
import numpy as np

class RuntimeEngine:

    saved_model_directory = './saved_models'
    input_data = '/home/miles/kaggle_tf_audio/data/test/audio'
    output_file = 'mrp_submission.csv'

    def run(self):
        cnt=0
        results = list()
        du = DataUtility()
        preprocessor = PreProcessor()
        raw_files = du.get_filenames(self.input_data)
        model = du.load_latest_model(self.saved_model_directory)
        classes = du.load_latest_category(self.saved_model_directory)

        saved_file = open("results.csv", 'w')
        saved_file.write('fname,label\n')

        for f in raw_files:
            full_path = "{0}/{1}".format(self.input_data, f)
            features = preprocessor.transform_audio_to_features(full_path)
            features = np.reshape(features, (99, 26, 1))
            features = np.expand_dims(features, axis=0)
            a = model.predict(features, batch_size=1, verbose=0)
            idx = classes[np.argmax(a)]
            if idx not in ['on','off','up','down','left','right','stop','go','yes','no']:
                idx = 'other'
            txt = "{0}, {1}\n".format(f, idx)
            cnt=cnt+1
            if cnt%1000==0:
                print("{0} of {1}:  {2}%".format(cnt, len(raw_files), int(cnt/len(raw_files)*100)))
            saved_file.write(txt)

if __name__ == '__main__':
    re = RuntimeEngine()
    re.run()