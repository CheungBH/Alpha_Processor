# %%
import itertools
import numpy as np
import matplotlib.pyplot as plt
import os


class Utils(object):
    def __init__(self):
        pass

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

    @staticmethod
    def plot_confusion_matrix(cm, classes, num, data_path,
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
        CM_dir = os.path.join(data_path, "graph/Confusion_Matrix")
        os.makedirs(CM_dir, exist_ok=True)
        plt.savefig(CM_dir + '/CM_' + str(num))
        plt.close()