from python_speech_features import mfcc
from python_speech_features import logfbank
import os
import scipy.io.wavfile as wav
import scipy.misc
import time
import shutil
from PIL import Image
import numpy as np


class PreProcessor:

    __version__ = '1.0.0'

    def __init__(self):
        print("Preprocessor initializing.  Version: {0}".format(self.__version__))

    def transform_audio_to_features(self, path):
        (rate, sig) = wav.read(path)
        filter_bank_features = logfbank(sig, rate, nfft=1600)
        if filter_bank_features.shape[0] < 99 or filter_bank_features.shape[1] < 26:
            print("Reshaping...")
            zeros = np.zeros((99, 26), dtype=np.int32)
            zeros[:filter_bank_features.shape[0], :filter_bank_features.shape[1]] = filter_bank_features
            return zeros
        else:
            return filter_bank_features


if __name__ == '__main__':
    print("\n\n\n")
    print("Preprocessor is a library and not intended to be run separately.")
    print("\n\n\n")