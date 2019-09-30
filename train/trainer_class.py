# %%
import IPython
import pandas as pd
import keras
import itertools
import numpy as np
import matplotlib.pyplot as plt
from train.LSTMmodels import ModelStorage
from utils.utils import Utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.callbacks import EarlyStopping,ModelCheckpoint
import os
from config import config

batchsize = config.batch_size
activation = config.activation
optimizer = config.optimizer
data_path = config.data_path
# %%

def RunNetwork(epo, dropout, networknum, class_name, Xvector, valpercent, timestep, iteration, data_info):
    timesteps = timestep
    epochs = epo
    batch_size = batchsize
    _dropout = dropout
    _activation = activation
    _optimizer = optimizer
    class_names = class_name  # 4 classes
    X_vector_dim = Xvector  # number of features or columns (pose)
    samples_path = os.path.join(data_path, "data.txt")
    labels_path = os.path.join(data_path, "label.txt")
    os.makedirs(os.path.join(data_path, str(iteration)), exist_ok=True)
    model_weights_path = os.path.join(data_path, str(iteration), 'pose_model.h5')

    X = np.loadtxt(samples_path, dtype="float")
    y = np.loadtxt(labels_path)

    X_vectors_per_sample = timesteps  # number of vectors per sample , 5 samples
    X_3D = Utils.samples_to_3D_array(X_vector_dim, X_vectors_per_sample, X)
    y_one_hot = Utils.convert_y_to_one_hot(y)
    y_vector_dim = y_one_hot.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X_3D, y_one_hot, test_size=valpercent, random_state=42)
    input_shape = (X_train.shape[1], X_train.shape[2])

    MS = ModelStorage(networknum, _dropout, X_vector_dim, _activation, input_shape,
                                               y_vector_dim)
    model, network_info = MS.get_model()

    model.compile(loss='categorical_crossentropy', optimizer=_optimizer, metrics=['accuracy'])

    class TrainingVisualizer(keras.callbacks.History):
        def on_epoch_end(self, epoch, logs={}):
            super(TrainingVisualizer, self).on_epoch_end(epoch, logs)
            IPython.display.clear_output(wait=True)

        # 生成TrainingVisualizer图片
        def on_train_end(self, epochs, ogs=None):
            axes = pd.DataFrame(self.history).plot()
            axes.axvline(x=max((val_acc, i) for i, val_acc in enumerate(self.history['val_acc']))[1])
            os.makedirs(os.path.join(data_path, "graph/TrainingVisualizer"), exist_ok=True)
            plt.savefig(os.path.join(data_path, 'graph/TrainingVisualizer/TV_' + str(iteration)))

    early_stopping = EarlyStopping(monitor='val_acc', patience=10)
    model_checkpoint = ModelCheckpoint(filepath=model_weights_path, monitor='val_loss', save_best_only=True)

    print('Training...')
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_test, y_test),
              callbacks=[TrainingVisualizer(), early_stopping, model_checkpoint])

    # print('early_stopping:',early_stopping)

    score, accuracy = model.evaluate(X_test, y_test,
                                     batch_size=batch_size)

    doc = open(os.path.join(data_path, 'result.txt'), 'a')
    print(iteration, file=doc)
    print('Test score: {:.3}'.format(score), file=doc)
    print('Test accuracy: {:.3}'.format(accuracy), file=doc)
    doc.close()

    y_pred = model.predict(X_test)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    Utils.plot_confusion_matrix(cnf_matrix, classes=class_names, num=iteration, data_path=data_path,
                          title='Confusion matrix, without normalization')

    docdes = open(os.path.join(data_path,'description.txt'), 'a')
    docdes.write(str(iteration) + "\n")
    docdes.write("Data source: {}\n".format(data_info))
    docdes.write("Network: {}\n".format(network_info))
    docdes.write("epochs: {}\n".format(epo))
    docdes.write("dropout: {}\n".format(dropout))
    docdes.write("Validation percentage: {}\n\n".format(dropout))
    docdes.close