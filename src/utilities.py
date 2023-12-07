import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from mlcm import mlcm
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix, classification_report


class Utilities:
    emotion_mapping = {0:"joy", 1:"desire", 2:"pride", 3:"agreement", 4:"surprise", 5:"love", 6:"confusion", 7:"anger", 8:"disgust", 9:"sadness", 10:"fear", 11:"optimism", 12:"disappointment", 13:"neutral",}

    TAG_RE = re.compile(r'\[[^]]+\]')

    def remove_tags(self, text):
        """
        Removes [] tags: replaces anything between opening and closing [] with empty space or with any words
        """
        return self.TAG_RE.sub('', text)
    
    def preprocess_text(self, sen, removeStopWords = True):
        '''Cleans text data up, leaving only 2 or more char long non-stepwords composed of A-Z & a-z only
        in lowercase'''
        
        sentence = sen.lower()

        # Remove placeholders like [NAME], [RELIGION], etc...
        sentence = self.remove_tags(sentence)

        # Remove punctuations and numbers
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)
       
        # Single character removal
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)  # When we remove apostrophe from the word "Mark's", the apostrophe is replaced by an empty space. Hence, we are left with single character "s" that we are removing here.

        # Remove multiple spaces
        sentence = re.sub(r'\s+', ' ', sentence)  # Next, we remove all the single characters and replace it by a space which creates multiple spaces in our text. Finally, we remove the multiple spaces from our text as well.

        if removeStopWords:# Remove Stopwords
            nltk.download("stopwords", quiet=True)
            pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
            sentence = pattern.sub('', sentence)

        return sentence
    
    def process_text_list(self, sentences, labels, removeStopWords=True):
        X = []
        Y = []
        for i,sen in enumerate(sentences):
            X.append(self.preprocess_text(sen, removeStopWords))
            if len(labels) > 0:
                Y.append([int(x) for x in labels[i].replace(' ', '').split(',')])
        return X,Y

    def one_hot_encode(self, Y):
        one_hot_encoded = np.zeros((len(Y), 14))
        for i, y in enumerate(Y):
            if type(y) is not list:                
                emotions = y.replace(' ', '').split(',')
                for emo_index in emotions:
                    one_hot_encoded[i][int(emo_index)] = 1
            else:                
                for emo_index in y:
                    one_hot_encoded[i][emo_index] = 1
        return one_hot_encoded
    
    def plot_confusion_matrix(self, true_lbl, pred_lbl, argMax=False):
        if argMax:
            cm = confusion_matrix(true_lbl.argmax(axis=1), pred_lbl.argmax(axis=1))            
            print("Classification Report")
            print(classification_report(true_lbl.argmax(axis=1), pred_lbl.argmax(axis=1)))
        else:
            conf_mat, normal_conf_mat = mlcm.cm(true_lbl, pred_lbl)
            cm = np.delete(conf_mat, 14, axis=1)
            cm = np.delete(cm, 14, axis=0)
        labels = list(self.emotion_mapping.values())
        cm_df = pd.DataFrame(cm, columns=labels, index=labels)
        plt.figure(figsize=(9,9))
        sns.heatmap(cm_df, annot=True, fmt=".1f")
        plt.ylabel('Actual Values')
        plt.xlabel('Predicted Values')
        plt.title("Confusion  Matrix")
        plt.show()        

    def plot_confusion_matrix_norm(self, true_lbl, pred_lbl, argMax=False):
        conf_mat, normal_conf_mat = mlcm.cm(true_lbl, pred_lbl)
        cm = np.delete(normal_conf_mat, 14, axis=1)
        cm = np.delete(cm, 14, axis=0)
        labels = list(self.emotion_mapping.values())
        cm_df = pd.DataFrame(cm, columns=labels, index=labels)
        plt.figure(figsize=(9,9))
        sns.heatmap(cm_df, annot=True, fmt=".1f")
        plt.ylabel('Actual Values')
        plt.xlabel('Predicted Values')
        plt.title("Confusion  Matrix")
        plt.show()


    