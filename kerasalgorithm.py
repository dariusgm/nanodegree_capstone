import numpy as np
import os

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
# For clean resetting: https://stackoverflow.com/questions/45063602/attempting-to-reset-tensorflow-graph-when-using-keras-failing


# Based on example code by https://keras.io/getting-started/sequential-model-guide/
class KerasAlgorithm:
    def __init__(self, drive_csv):
        self.model = Sequential()
        self.batch_size = 65536
        self.valid_model = None
        self.drive_csv = drive_csv
        
    def __str__(self):
        return self.model.to_json()

    def fit(self, X_train, y_train, X_valid, y_valid):
        self.model.add(Dense(len(X_train.columns), input_dim=len(X_train.columns), activation='relu'))
        
        # 0.9*len, 0.8*len ... until 0.1
        for i in range(9, 0, 1):
            dense_size = min(2, int(len(X_train.columns) * (i/10)))
            if dense_size == 2:
                # Skip iteration, as network is getting to small
                continue
            else:
                self.model.add(Dense(dense_size, activation='relu'))

        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',
               # used: https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/
               # for the correct optimizer 
              optimizer='adam',
              metrics=['binary_accuracy'])
        self.checkpointer = ModelCheckpoint(filepath=os.path.join('keras_models', '{}.hdf5'.format(self.drive_csv)), 
                               verbose=0, save_best_only=True)
        
        # From https://www.quora.com/How-can-I-stop-training-in-Keras-if-it-does-not-improve-for-two-consecutive-epochs
        self.early_stopping_monitor = EarlyStopping(patience=3, min_delta=0.00001)

        history = self.model.fit(X_train,
                                 y_train,
                                 epochs=10,
                                 batch_size=self.batch_size, 
                                 callbacks=[
                                     self.checkpointer,
                                     self.early_stopping_monitor],
                                 verbose=0,
                                 validation_data=(X_valid, y_valid))
        # Using 0.1, as one drive have a val_binary_accuracy of 0.02, that still can't generate any predictions
        if np.array(history.history['val_binary_accuracy']).sum() > 0.1:
            self.valid_model = True
        else:
            self.valid_model = False
        return self

    def save(self, filename):
            self.model.save(filename)

        
    def predict(self, X_test):
        if not self.valid_model:
            # using: https://stackoverflow.com/questions/5891410/numpy-array-initialization-fill-with-identical-values
            # when the model have not enough datapoints, we can't train it and return only false for each element in the validation set
            # this also makes sure that the fbeta score can be correctly calculated
            # otherwise you get nan's that aren't usefull at all.
            return np.full((1, len(X_test)), 0, dtype=int)[0]
        else:
            prediction = self.model.predict(X_test)
            return prediction