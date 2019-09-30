from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, TimeDistributed, Activation
from keras.layers import LSTM


class ModelStorage(object):
    def __init__(self, num, _dropout, X_vector_dim, _activation, input_shape, y_vector_dim):
        self.number = num
        if self.number == 0:
            self.model = Sequential()
            self.model.add(TimeDistributed(Dense(X_vector_dim, activation=_activation), input_shape=input_shape))
            self.model.add(Dropout(_dropout))
            self.model.add(TimeDistributed(Dense(X_vector_dim * 2, activation=_activation)))  # (5, 80)
            self.model.add(Dropout(_dropout))
            self.model.add(TimeDistributed(Dense(X_vector_dim, activation=_activation)))  # (5, 40)
            self.model.add(Dropout(_dropout))
            self.model.add(TimeDistributed(Dense(int(X_vector_dim / 2), activation=_activation)))  # (5, 20)
            self.model.add(Dropout(_dropout))
            self.model.add(TimeDistributed(Dense(int(X_vector_dim / 4), activation=_activation)))  # (5, 10)
            self.model.add(Dropout(_dropout))
            self.model.add(LSTM(int(X_vector_dim / 4), dropout=_dropout, recurrent_dropout=_dropout))
            self.model.add(Dense(y_vector_dim, activation='softmax'))
            self.info = '*2_*1_/2_/4'
        elif self.number == 1:
            self.model = Sequential()
            self.model.add(TimeDistributed(Dense(X_vector_dim, activation=_activation), input_shape=input_shape))
            self.model.add(Dropout(_dropout))
            self.model.add(TimeDistributed(Dense(X_vector_dim * 2, activation=_activation)))  # (5, 80)
            self.model.add(Dropout(_dropout))
            self.model.add(TimeDistributed(Dense(X_vector_dim * 2, activation=_activation)))  # (5, 80)
            self.model.add(Dropout(_dropout))
            self.model.add(TimeDistributed(Dense(X_vector_dim, activation=_activation)))  # (5, 40)
            self.model.add(Dropout(_dropout))
            self.model.add(TimeDistributed(Dense(int(X_vector_dim / 2), activation=_activation)))  # (5, 20)
            self.model.add(Dropout(_dropout))
            self.model.add(TimeDistributed(Dense(int(X_vector_dim / 4), activation=_activation)))  # (5, 10)
            self.model.add(Dropout(_dropout))
            self.model.add(LSTM(int(X_vector_dim / 4), dropout=_dropout, recurrent_dropout=_dropout))
            self.model.add(Dense(y_vector_dim, activation='softmax'))
            self.info = '*2_*2_*1_/2_/4'
        elif self.number == 2:
            self.model = Sequential()
            self.model.add(TimeDistributed(Dense(X_vector_dim, activation=_activation), input_shape=input_shape))
            self.model.add(Dropout(_dropout))
            self.model.add(TimeDistributed(Dense(X_vector_dim * 2, activation=_activation)))  # (5, 80)
            self.model.add(Dropout(_dropout))
            self.model.add(TimeDistributed(Dense(X_vector_dim, activation=_activation)))  # (5, 40)
            self.model.add(Dropout(_dropout))
            self.model.add(TimeDistributed(Dense(int(X_vector_dim / 2), activation=_activation)))  # (5, 20)
            self.model.add(Dropout(_dropout))
            self.model.add(LSTM(int(X_vector_dim / 2), dropout=_dropout, recurrent_dropout=_dropout))
            self.model.add(Dense(y_vector_dim, activation='softmax'))
            self.info = '_*2_*1_/2'
        elif self.number == 3:
            self.model = Sequential()
            self.model.add(TimeDistributed(Dense(X_vector_dim, activation=_activation), input_shape=input_shape))
            self.model.add(Dropout(_dropout))
            self.model.add(TimeDistributed(Dense(X_vector_dim * 2, activation=_activation)))  # (5, 80)
            self.model.add(Dropout(_dropout))
            self.model.add(TimeDistributed(Dense(X_vector_dim * 4, activation=_activation)))  # (5, 80)
            self.model.add(Dropout(_dropout))
            self.model.add(TimeDistributed(Dense(X_vector_dim * 2, activation=_activation)))  # (5, 80)
            self.model.add(Dropout(_dropout))
            self.model.add(TimeDistributed(Dense(X_vector_dim, activation=_activation)))  # (5, 40)
            self.model.add(Dropout(_dropout))
            self.model.add(TimeDistributed(Dense(int(X_vector_dim / 2), activation=_activation)))  # (5, 20)
            self.model.add(Dropout(_dropout))
            self.model.add(TimeDistributed(Dense(int(X_vector_dim / 4), activation=_activation)))  # (5, 10)
            self.model.add(Dropout(_dropout))
            self.model.add(LSTM(int(X_vector_dim / 4), dropout=_dropout, recurrent_dropout=_dropout))
            self.model.add(Dense(y_vector_dim, activation='softmax'))
            self.info = '*2_*4_*2_*1_/2_/4'
        elif self.number == 4:
            self.model = Sequential()
            self.model.add(TimeDistributed(Dense(X_vector_dim, activation=_activation), input_shape=input_shape))
            self.model.add(Dropout(_dropout))
            self.model.add(TimeDistributed(Dense(X_vector_dim * 2, activation=_activation)))  # (5, 80)
            self.model.add(Dropout(_dropout))
            self.model.add(TimeDistributed(Dense(int(X_vector_dim / 2), activation=_activation)))  # (5, 20)
            self.model.add(Dropout(_dropout))
            self.model.add(LSTM(int(X_vector_dim / 2), dropout=_dropout, recurrent_dropout=_dropout))
            self.model.add(Dense(y_vector_dim, activation='softmax'))
            self.info = '_*2_/2'
        else:
            raise ValueError("Number input is too big")

    def get_model(self):
        return self.model, self.info


