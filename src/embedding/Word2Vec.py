from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import numpy as np
from gensim.models import Word2Vec
from nltk import word_tokenize, sent_tokenize
import nltk 
from ..utilities import Utilities


class Word_2_Vec:

    util = Utilities()
    nltk.download('punkt', quiet=True)

    def __init__(self, X_train, vectorSize=100) -> None:
        self.vectorSize = vectorSize
        self.tokenize_texts(X_train)

    def cleanSentenceToWords(self, sentences):
        sent = [ ' '.join(sent_tokenize(sentence)).lower() for sentence in sentences]
        tokens = [' '.join(word_tokenize(sentence, preserve_line=True)) for sentence in sent]  
        return tokens

    def getTextToSequence(self, sentences, maxlen = 100):
        sequences = self.word_tokenizer.texts_to_sequences(sentences)
        return pad_sequences(sequences, maxlen=maxlen, padding='post')

    def tokenize_texts(self, sentences, epochs = 20):

        self.word_tokenizer = Tokenizer()

        self.tokens = ' '.join(self.cleanSentenceToWords(sentences))

        self.word_tokenizer.fit_on_texts(self.tokens)
        # self.vocab_length = len(set(self.tokens)) + 2

        # self.tokens = [x.split() for x in self.tokens]

        self.word2vec = Word2Vec(
            sentences=self.tokens, vector_size=self.vectorSize, min_count=1, sg=1)
        
        self.word2vec.train(self.tokens, total_examples=len(sentences), epochs=epochs)

        self.word2vec.save("word2vec.model")

        # print(f"Most Similar WV for 'love';\n{self.word2vec.wv.most_similar('love')}")

        self.vocab_length = len(self.word2vec.wv)
        self.embedding_matrix = self.word2vec.wv.vectors
