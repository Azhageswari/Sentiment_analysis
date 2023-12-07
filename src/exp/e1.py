from ..utilities import Utilities
from src.embedding.glove import GloVE
from src.models.SNN import SNN
from sklearn.model_selection import train_test_split



class Exp1:
    util = Utilities()

    def run(self, emotions, randomState, withStopwords=True):
        sentences = list(emotions['text'])
        labels = list(emotions['sentiments'])
        X,Y = self.util.process_text_list(sentences, labels, removeStopWords=(not withStopwords))
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=randomState)
        glove = GloVE(X_train)

        X_train_vec = glove.getTextToSequence(X_train) 
        X_test_vec = glove.getTextToSequence(X_test)
        #scatter_plot
        print("Scatter Plot for using with Stopwords range between 100 to 200")
        glove.scatter_plot_vectors_range(range_from=100, range_to=200)

        #One hot encodding the labels
        Y_train_one_hot_encoded = self.util.one_hot_encode(y_train)
        Y_test_one_hot_encoded = self.util.one_hot_encode(y_test)

        print("Embedding Layers of SNN")
        snn = SNN(X_train_vec, Y_train_one_hot_encoded, glove.vocab_length, glove.embedding_matrix, trainable=False)
        snn.train()
        snn.evaluate(X_test_vec, Y_test_one_hot_encoded)