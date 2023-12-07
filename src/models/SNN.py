# Simple Neural Network
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, GlobalMaxPooling1D, Embedding, BatchNormalization
import matplotlib.pyplot as plt
import numpy as np
from ..utilities import Utilities


class SNN:

    def __init__(self, X, Y, vocab_length, embedding_matrix, trainable=False):
        self.X = X
        self.Y = Y
        self.vocab_length = vocab_length
        self.embedding_matrix = embedding_matrix
        self.trainable = trainable
        self.model = self.create_model()

    def create_model(self):
        snn_model = Sequential()
        embedding_layer = Embedding(self.vocab_length, 100, weights=[self.embedding_matrix], input_length=100, trainable=self.trainable)
        snn_model.add(embedding_layer)
        snn_model.add(Flatten())
        # snn_model.add(BatchNormalization())#
        snn_model.add(Dense(128, activation='relu'))
        snn_model.add(BatchNormalization())#
        snn_model.add(Dense(64, activation='relu'))#
        snn_model.add(BatchNormalization())#
        snn_model.add(Dense(14, activation='softmax'))
        snn_model.compile(optimizer='adam',
                          loss='categorical_crossentropy', metrics=['acc'])
        print(snn_model.summary())
        return snn_model

    def train(self):
        # Model training
        self.model_history = self.model.fit(self.X, self.Y, batch_size=128, epochs=6,
                                            verbose=1, validation_split=0.2)

    def evaluate(self, X, Y):
        # Predictions on the Test Set
        score,acc = self.model.evaluate(X, Y, verbose=1)
        print(f"Test Score: {score}")
        print(f"Test Accuracy: {acc}")
        self.plotPerformance()
        self.plotConfusion(X, Y, argMax=True)

        return score,acc

    def plotConfusion(self, X, Y, normalize=False, argMax=False):
        ypred = self.model(X)
        util = Utilities()
        if normalize:
            util.plot_confusion_matrix_norm(Y, np.array(ypred))
        else:
            util.plot_confusion_matrix(Y, np.array(ypred), argMax=argMax)

    def plotPerformance(self):
        print(self.model_history.history)
        plt.plot(self.model_history.history['acc'])
        plt.plot(self.model_history.history['val_acc'])

        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(self.model_history.history['loss'])
        plt.plot(self.model_history.history['val_loss'])

        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
