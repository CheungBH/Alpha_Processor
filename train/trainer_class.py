import IPython
import pandas as pd
import keras
import itertools
import numpy as np
import matplotlib.pyplot as plt
import ModelStorage

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, TimeDistributed, Activation
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from kerasify import export_model
from keras.models import model_from_json
from keras.callbacks import EarlyStopping,ModelCheckpoint
import os
from config import config

batch_size = config.batch_size
activation = config.activation
optimizer = config.optimizer


class TrainingVisualizer(keras.callbacks.History):
    def on_epoch_end(self, epoch, logs={}):
        super(TrainingVisualizer, self).on_epoch_end(epoch, logs)
        IPython.display.clear_output(wait=True)

    # 生成TrainingVisualizer图片
    def on_train_end(self, epoch, ogs=None):
        axes = pd.DataFrame(self.history).plot()
        axes.axvline(x=max((val_acc, i) for i, val_acc in enumerate(self.history['val_acc']))[1])
        if os.path.isdir("graph/TrainingVisualizer") == True:
            pass
        else:
            os.makedirs("graph/TrainingVisualizer")
        plt.savefig('graph/TrainingVisualizer/TV_' + str(iteration))


class trainer(object):
    def __init__(self, epoch, dropout, networknum, class_name, Xvector, valpercent, timestep, iteration, dataInfo):
        self.timestep = timestep
        ## POSE - val_acc: 98%
        self.epoch = epoch
        self.batch_size = batch_size
        self._dropout = dropout
        self.network_num = networknum
        self.val_ratio = valpercent
        self.data_info = dataInfo
        self._activation = activation
        self._optimizer = optimizer
        self.class_name = class_name  # 4 classes
        self.X_vector_dim = Xvector  # number of features or columns (pose)
        self.samples_path = "data.txt"  # 311 files with 10 frames' human-pose estimation keypoints(10*18)
        self.labels_path = "label.txt"  # 311 files' labels, 3 classes in total
        os.makedirs(str(iteration), exist_ok=True)
        self.model_path = str(iteration) + '/pose.model'
        self.json_model_path = str(iteration) + '/pose_model.json'
        self.model_weights_path = str(iteration) + '/pose_model.h5'

    @staticmethod
    def samples_to_3D_array(_vector_dim, _vectors_per_sample, _X):
        X_len = len(_X)
        result_array = []
        for sample in range(0, X_len):  # should be the 311 samples?
            sample_array = []
            for vector_idx in range(0, _vectors_per_sample):
                start = vector_idx * _vector_dim
                end = start + _vector_dim
                sample_array.append(_X[sample][start:end])
            result_array.append(sample_array)
        return np.asarray(result_array)

    @staticmethod
    def convert_y_to_one_hot(_y):  # one hot encoding simply means : red --> 0 , green --> 1 , blue --> 2
        _y = np.asarray(_y, dtype=int)
        b = np.zeros((_y.size, _y.max() + 1))
        b[np.arange(_y.size), _y] = 1
        return b



    def auto_running(self):
        X = np.loadtxt(self.samples_path, dtype="float")
        y = np.loadtxt(self.labels_path)

        X_vectors_per_sample = self.timestep  # number of vectors per sample , 5 samples
        X_3D = self.samples_to_3D_array(self.X_vector_dim, X_vectors_per_sample, X)

        y_one_hot = self.convert_y_to_one_hot(y)
        y_vector_dim = y_one_hot.shape[1]

        X_train, X_test, y_train, y_test = train_test_split(X_3D, y_one_hot, test_size=valpercent, random_state=42)
        input_shape = (X_train.shape[1], X_train.shape[2])

        model,NetworkInfo=ModelStorage.GetModel(self.network_num, self._dropout, self.X_vector_dim, self._activation, input_shape, y_vector_dim)
        model.compile(loss='categorical_crossentropy', optimizer=self._optimizer, metrics=['accuracy'])

