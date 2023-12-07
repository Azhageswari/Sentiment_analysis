from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class GloVE:

    def __init__(self, X_train) -> None:
        self.word_tokenizer = Tokenizer()
        self.word_tokenizer.fit_on_texts(X_train)
        self.loadGloveEmbeddings()
        
    def getTextToSequence(self, sentences):
        word_vec = self.word_tokenizer.texts_to_sequences(sentences)
        return pad_sequences(word_vec, padding='post', maxlen=100)
    
    def loadGloveEmbeddings(self):
        self.embeddings_dictionary = dict()
        glove_file = open('glove/a2_glove.6B.100d.txt', encoding="utf8")

        for line in glove_file:
            records = line.split()
            word = records[0]
            vector_dimensions = np.asarray(records[1:], dtype='float32')
            self.embeddings_dictionary[word] = vector_dimensions
        glove_file.close()

        # Adding 1 to store dimensions for words for which no pretrained word embeddings exist
        self.vocab_length = len(self.word_tokenizer.word_index) + 1 
        
        # Create Embedding Matrix having 100 columns 
        # Containing 100-dimensional GloVe word embeddings for all words in our corpus.
        
        self.embedding_matrix = np.zeros((self.vocab_length, 100))
        for word, index in self.word_tokenizer.word_index.items():
            embedding_vector = self.embeddings_dictionary.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[index] = embedding_vector
    
    def scatter_plot_vectors_range(self, range_from=0, range_to=700):
        distri = TSNE(n_components=2)
        words = list(self.word_tokenizer.word_index.keys())
        vectors = self.embedding_matrix
        # words = list(self.embeddings_dictionary.keys())
        # vectors = [self.embeddings_dictionary[word] for word in words]
        y = distri.fit_transform(np.array(vectors[range_from:range_to]))
        plt.figure(figsize=(14,8))
        plt.scatter(y[:, 0],y[:,1])
        for label,x,y in zip(words,y[:, 0],y[:,1]):
            plt.annotate(label,xy=(x,y),xytext=(0,0),textcoords='offset points')
        plt.show()