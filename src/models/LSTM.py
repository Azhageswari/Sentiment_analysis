# Neural Network architecture
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, GlobalMaxPooling1D, Embedding, LSTM, BatchNormalization
import matplotlib.pyplot as plt
import numpy as np
from ..utilities import Utilities


class LSTMClassifier:

    def __init__(self, X, Y, vocab_length, embedding_matrix, trainable=False):
        self.X = X
        self.Y = Y
        self.vocab_length = vocab_length
        self.embedding_matrix = embedding_matrix
        self.trainable = trainable
        self.model = self.create_model()

    def create_model(self):

        lstm_model = Sequential()
        embedding_layer = Embedding(self.vocab_length, 100, weights=[
                                    self.embedding_matrix], input_length=100, trainable=self.trainable)
        lstm_model.add(embedding_layer)
        lstm_model.add(LSTM(128)) 
        #above i need to add 
        # lstm_model.add(BatchNormalization())        
        # lstm_model.add(LSTM(128, dropout=0.2, activation='relu'))#
        lstm_model.add(Dense(64, activation='relu'))
        lstm_model.add(BatchNormalization())
        lstm_model.add(Dense(14, activation='softmax'))
        print(lstm_model.summary())
        lstm_model.compile(
            optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        return lstm_model

    def train(self):
        # Model Training
        self.model_history = self.model.fit(
            self.X, self.Y, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

    def evaluate(self, X, Y):
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
