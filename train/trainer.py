# %%
import IPython
import pandas as pd
import keras
import itertools
import numpy as np
import matplotlib.pyplot as plt
from train import ModelStorage

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.models import model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import EarlyStopping,ModelCheckpoint
import os
from config import config

batchsize = config.batch_size
activation = config.activation
optimizer = config.optimizer
data_path = config.data_path
# %%

def RunNetwork(epo, dropout, networknum, class_name, Xvector, valpercent, timestep, iteration, dataInfo):
    timesteps = timestep
    ## POSE - val_acc: 98%
    epochs = epo
    batch_size = batchsize
    _dropout = dropout
    _activation = activation
    _optimizer = optimizer
    class_names = class_name  # 4 classes
    X_vector_dim = Xvector  # number of features or columns (pose)
    samples_path = "data.txt"  # 311 files with 10 frames' human-pose estimation keypoints(10*18)
    labels_path = "label.txt"  # 311 files' labels, 3 classes in total
    os.makedirs(os.path.join(data_path, str(iteration)), exist_ok=True)
    model_path = data_path + "/" +str(iteration) + '/pose.model'
    json_model_path = data_path + "/" + str(iteration) + '/pose_model.json'
    model_weights_path = data_path + "/" + str(iteration) + '/pose_model.h5'

    X = np.loadtxt(samples_path, dtype="float")
    y = np.loadtxt(labels_path)

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

    X_vectors_per_sample = timesteps  # number of vectors per sample , 5 samples
    X_3D = samples_to_3D_array(X_vector_dim, X_vectors_per_sample, X)

    def convert_y_to_one_hot(_y):  # one hot encoding simply means : red --> 0 , green --> 1 , blue --> 2
        _y = np.asarray(_y, dtype=int)
        b = np.zeros((_y.size, _y.max() + 1))
        b[np.arange(_y.size), _y] = 1
        return b

    y_one_hot = convert_y_to_one_hot(y)
    y_vector_dim = y_one_hot.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(X_3D, y_one_hot, test_size=valpercent, random_state=42)
    input_shape = (X_train.shape[1], X_train.shape[2])

    model, NetworkInfo = ModelStorage.GetModel(networknum, _dropout, X_vector_dim, _activation, input_shape,
                                               y_vector_dim)
    model.compile(loss='categorical_crossentropy', optimizer=_optimizer, metrics=['accuracy'])

    class TrainingVisualizer(keras.callbacks.History):
        def on_epoch_end(self, epoch, logs={}):
            super(TrainingVisualizer, self).on_epoch_end(epoch, logs)
            IPython.display.clear_output(wait=True)

        # 生成TrainingVisualizer图片
        def on_train_end(self, epoch, ogs=None):
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

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        if os.path.isdir("graph/Confusion_Matrix") == True:
            pass
        else:
            os.makedirs("graph/Confusion_Matrix")
        plt.savefig('graph/Confusion_Matrix/CM_' + str(iteration))
        plt.close()

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    docdes = open('description.txt', 'a')
    docdes.write(str(iteration) + "\n")
    docdes.write("Data source: " + dataInfo + "\n")
    docdes.write("Network: " + NetworkInfo + "\n")
    docdes.write("epochs: " + str(epo) + "\n")
    docdes.write("dropout: " + str(dropout) + "\n")
    docdes.write("Validation percentage: " + str(valpercent) + "\n")
    docdes.write('\n')
    docdes.close