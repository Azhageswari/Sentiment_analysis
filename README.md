Sentiment Analysis for Business Improvement

This repository contains code and resources for sentiment analysis aimed at understanding customer and public feedback to improve business quality. The project employs various methodologies such as data preprocessing, embedding techniques, algorithm selection, hyperparameter tuning, test/train splitting, and accuracy evaluation to categorize sentiments and enhance the business model.

Overview
The primary objective of this project is to utilize sentiment analysis on customer reviews and feedback to identify criticisms and areas for improvement within the business model. The repository contains code that demonstrates different approaches to sentiment analysis using machine learning techniques, focusing on text data obtained from diverse sources including social media.

Contents
data_preprocessing/: Code and scripts for preprocessing textual data (e.g., removing special characters, converting to lowercase, stop-word removal)
embedding/: Implementation of various embedding techniques such as Word Embedding, Bag of Words, Word2Vec, and GloVe
model_training/: Scripts for training models including Simple Neural Networks, CNNs, and LSTMs
hyperparameter_tuning/: Techniques and scripts for hyperparameter optimization (e.g., learning rate, batch size, number of epochs)
evaluation/: Code to evaluate model performance metrics including accuracy, confusion matrix, and classification report
data_analysis/: Tools and notebooks for data analysis and visualization

Data Sources
The project utilizes the "GoEmotions" dataset derived from Reddit, encompassing 27 labels and 58K metadata. The dataset has been distilled from 54K metadata, considering 13 top sentiments along with neutral labels. The README file within the data folder provides detailed information on the data and its structure. If you need of this dataset you can email me directly I can share train and test split.

Methodology
The methodology involves experimenting with different data preprocessing strategies, vectorization techniques (e.g., Word2Vec, GloVe), learning rates, and algorithms (e.g., CNNs). The project discusses the advantages of removing stopwords, the utility of GloVe for custom word embeddings, and the scalability and feature extraction capabilities of CNNs.



Results
The project concludes that the Convolutional Neural Network (CNN) demonstrates the best performance for multi-class sentiment classification, achieving the highest model score after comprehensive tuning.

Contributors
Azhagesawri
Jeevanantham
Ezhil Priyadharshini
Yaswanth
Aiswarya

Feel free to contribute, suggest improvements, or report issues by forking the repository, making changes, and creating a pull request. For any queries or suggestions, contact alaguvarsha23@gmail.com.
