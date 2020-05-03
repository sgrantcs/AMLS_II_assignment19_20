# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 20:21:39 2020

@author: svetl
"""

import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
#import seaborn as sns
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.classify import NaiveBayesClassifier
from Processing import process_tweets

def run_task_A():
    # Data preprocessing
    data = pd.read_table('SemEval2017-task4-dev.subtask-A.english.INPUT.txt', index_col=False, header =0, sep='\t', names=['ID','label','tweet'])
    process_tweets(data)
    #print(data.head(10))

    #Create bag-of-words feature
    bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
    # bag-of-words feature matrix
    bow = bow_vectorizer.fit_transform(data['tidy_tweet'])
    #print(bow)

    tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
    # TF-IDF feature matrix
    tfidf = tfidf_vectorizer.fit_transform(data['tidy_tweet'])
    #print(tfidf)

    #Build model for Bag of Words 

    #from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score

    train_bow = bow[:20632,:]
    test_bow = bow[20632:,:]
    #change data labels into categorical. 
    data['n_label'] = data['label'].str.replace("positive", "1")
    data['n_label'] = data['n_label'].str.replace("negative", "-1")
    data['n_label'] = data['n_label'].str.replace("neutral", "0")
    #print(data['n_label'])
    
    # splitting data into training and validation set (70:30)n_label
    xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, data['n_label'], random_state=42, test_size=0.3)
    # Train a Naive Bayes Multinomial classifier
    classifier = MultinomialNB()
    classifier.fit(xtrain_bow, ytrain)
    # making predictions on the testing set 
    y_pred = classifier.predict(xvalid_bow) 
    #classifier = NaiveBayesClassifier.train(train_bow)
    #print("\nAccuracy of the classifier:"), nltk.classify.util.accuracy(classifier, test_bow)
    from sklearn import metrics 
    print("Multinomial Task A Naive Bayes model accuracy(in %):", metrics.accuracy_score(yvalid, y_pred)*100)
    print("Multinomial Task A Naive Bayes model precision(in %):", metrics.precision_score(yvalid, y_pred, average='macro')*100)
    print("Multinomial Task A Naive Bayes model Recall score (in %):", metrics.recall_score(yvalid, y_pred, average='macro')*100)
    print("Multinomial Task A Naive Bayes model F1 score (in %):", metrics.f1_score(yvalid, y_pred, average='macro')*100)

