import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from .utilities import Utilities


class AnalyzeData:
    util = Utilities()

    def plot_label_frequency(self, emotions, excel_reduced=False) -> None:
       if excel_reduced:
        emotions = emotions.sort_values('sentiments')
        emotions.sentiments = emotions.sentiments.map(self.util.emotion_mapping)
        sn.countplot(y=emotions['sentiments'], data=emotions)
        plt.xticks(rotation=90)
       else:
        self.modified_df = emotions.assign(unique_sentiments=emotions.sentiments.str.split(','))
        self.modified_df = self.modified_df.explode('unique_sentiments')
        self.modified_df['unique_sentiments'] = self.modified_df.unique_sentiments.apply(lambda x: int(x.replace(' ', '')))
        self.modified_df = self.modified_df.sort_values('unique_sentiments')
        self.modified_df.unique_sentiments = self.modified_df.unique_sentiments.map(self.util.emotion_mapping)
        sn.countplot(y=self.modified_df['unique_sentiments'], data=self.modified_df)
        plt.xticks(rotation=90)


    def plot_correlation(self, Y):
        corr_matrix = pd.DataFrame(Y,columns=list(self.util.emotion_mapping.values()))
        sn.heatmap(corr_matrix.corr(), fmt=".1f", cmap='Greens', annot=True)
        plt.title("lables class Correlation")       
