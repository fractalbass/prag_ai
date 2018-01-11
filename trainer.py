from data_utility import DataUtility
from sklearn.model_selection import train_test_split
from time import time
from models import Models
from keras import callbacks as CB
from keras import backend as K
import numpy as np
from keras.utils import np_utils
from matplotlib import pyplot

class Trainer:

    du = DataUtility()
    models = Models()
    input_shape = None

    saved_models_directory = "./saved_models"
    training_root_directory = "/home/miles/kaggle_speech_data"

    img_width, img_height = 99, 26
    epochs = 2000
    batch_size = 32

    def __init__(self):
        if K.image_data_format() == 'channels_first':
            self.input_shape = (1, self.img_width, self.img_height)
        else:
            self.input_shape = (self.img_width, self.img_height, 1)

    def train(self):
        print("Training...")
        start_time = time()

        X, Y, ppv = self.du.load_npz_training_data(self.training_root_directory)
        class_set, Y_vector = self.du.get_vectorized_data(Y)

        x_train, y_train, x_test, y_test = train_test_split(X, Y_vector, test_size=0.1, random_state=42)

        # x_test = np_utils.to_categorical(x_test, len(class_set))
        # y_test = np_utils.to_categorical(y_test, len(class_set))

        x_train = np.expand_dims(x_train, axis=3)
        y_train = np.expand_dims(y_train, axis=3)


        model = self.models.get_av_blog_model_2(self.input_shape, len(class_set))

        history = model.fit(x=x_train, y=x_test, validation_data=(y_train, y_test),
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=1)

        end_time = time()
        print("Training complete.  Total training time: {0}".format(end_time-start_time))
        self.du.save_model(self.saved_models_directory, ppv, model)
        self.du.save_categories(self.saved_models_directory, ppv, class_set)
        print("Model and classes saved.")

        pyplot.plot(history.history['acc'])
        pyplot.plot(history.history['val_acc'])
        pyplot.title("Model Accuracy")
        pyplot.ylabel("loss")
        pyplot.xlabel("epoch")
        pyplot.legend(['train', 'test'], loc='upper left')
        pyplot.show()

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
