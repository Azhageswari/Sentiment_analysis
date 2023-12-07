# Neural Network architecture
from keras.layers import Conv1D
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, GlobalMaxPooling1D, Embedding, Conv1D, BatchNormalization
import matplotlib.pyplot as plt
import numpy as np
from ..utilities import Utilities


class CNN:

    def __init__(self, X, Y, vocab_length, embedding_matrix, trainable=False, lr=None):
        self.X = X
        self.Y = Y
        self.lr = lr
        self.vocab_length = vocab_length
        self.embedding_matrix = embedding_matrix
        self.trainable = trainable
        self.model = self.create_model()

    def create_model(self):
        model = Sequential()
        embedding_layer = Embedding(self.vocab_length, 100, weights=[
                                    self.embedding_matrix], input_length=100, trainable=self.trainable)
        model.add(embedding_layer)
        model.add(Conv1D(128, 5, activation='relu'))
        # model.add(BatchNormalization())    
        model.add(Conv1D(256, 5, activation='relu'))
        model.add(GlobalMaxPooling1D())
        # model.add(Flatten())
        # model.add(Dense(25600, activation='relu'))
        model.add(Dense(128, activation='relu'))    
        # model.add(BatchNormalization())    
        model.add(Dense(14, activation='softmax'))
        # Model compiling
        print(model.summary())
        if self.lr:
            optimizer = Adam(learning_rate=self.lr)
        else:
            optimizer = 'adam'
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy', metrics=['acc'])
        return model

    def train(self):
        # Model training
        self.model_history = self.model.fit(
            self.X, self.Y, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

    def evaluate(self, X, Y):
        # Predictions on the Test Set
        # Predictions on the Test Set
        score,acc = self.model.evaluate(X, Y, verbose=1)
        print(f"Test Score: {score}")
        print(f"Test Accuracy: {acc}")
        self.plotperformance()
        self.plotConfusion(X, Y, argMax=True)

    # def plotConfusion(self, X, Y):
    #     ypred = self.model(X)
    #     Utilities().plot_confusion_matrix(Y, np.array(ypred))

    def plotConfusion(self, X, Y, normalize=False, argMax=False):
        ypred = self.model(X)
        util = Utilities()
        if normalize:
            util.plot_confusion_matrix_norm(Y, np.array(ypred))
        else:
            util.plot_confusion_matrix(Y, np.array(ypred), argMax=argMax)

    def plotperformance(self):
        # Model Performance Charts
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
