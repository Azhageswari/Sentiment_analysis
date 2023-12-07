from ..utilities import Utilities
from src.embedding.glove import GloVE
from src.embedding.Word2Vec import Word_2_Vec
from src.models.SNN import SNN
from sklearn.model_selection import train_test_split
from src.models.CNN import CNN
from src.models.LSTM import LSTMClassifier




class Exp3:
    util = Utilities()

    def run(self, emotions, randomState, vec:str = 'glove', net:str = 'SNN'):
        sentences = list(emotions['text'])
        labels = list(emotions['sentiments'])
        X,Y = self.util.process_text_list(sentences, labels, removeStopWords=True)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=randomState)

        if vec == 'glove':
            vectorizer = GloVE(X_train)
            print("Scatter Plot for using with Stopwords range between 100 to 200")
            vectorizer.scatter_plot_vectors_range(range_from=100, range_to=200)
        else :
            vectorizer = Word_2_Vec(X_train)


        X_train_vec = vectorizer.getTextToSequence(X_train) 
        X_test_vec = vectorizer.getTextToSequence(X_test)
        #scatter_plot

        #One hot encodding the labels
        Y_train_one_hot_encoded = self.util.one_hot_encode(y_train)
        Y_test_one_hot_encoded = self.util.one_hot_encode(y_test)

        if net == 'SNN':
            print("Embedding Layers of SNN")
            snn = SNN(X_train_vec, Y_train_one_hot_encoded, vectorizer.vocab_length, vectorizer.embedding_matrix, trainable=True)
            snn.train()
            snn.evaluate(X_test_vec, Y_test_one_hot_encoded)
        elif net == 'CNN':
            cnn = CNN(X_train_vec, Y_train_one_hot_encoded, vectorizer.vocab_length, vectorizer.embedding_matrix, trainable=True)
            cnn.train()
            cnn.evaluate(X_test_vec, Y_test_one_hot_encoded)
        elif net == 'LSTM':
            lstm = LSTMClassifier(X_train_vec, Y_train_one_hot_encoded, vectorizer.vocab_length, vectorizer.embedding_matrix, trainable=True)
            lstm.train()
            lstm.evaluate(X_test_vec, Y_test_one_hot_encoded)