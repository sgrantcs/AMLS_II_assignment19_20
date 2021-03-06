
import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import string
import nltk
from nltk.stem.porter import *
#from nltk.classify import NaiveBayesClassifier
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics 
#Import SKlearn evaluation metrics functions
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from collections import Counter


data = pd.read_table('SemEval2017-task4-dev.subtask-BD.english.INPUT.txt', index_col=False, header =0, sep='\t', names=['ID','topic','label','tweet'])
#print(data.dtypes)
#print(data.head(10))

def run_task_D():
    # Data preprocessing
    data = pd.read_table('SemEval2017-task4-dev.subtask-A.english.INPUT.txt', index_col=False, header =0, sep='\t', names=['ID','label','tweet'])
    process_tweets(data)
    #print(data.head(10))

    #Create bag-of-words feature
    bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
    # bag-of-words feature matrix
    bow = bow_vectorizer.fit_transform(data['tidy_tweet'])
    #print(bow)


    train_bow = bow[:20632,:]
    test_bow = bow[20632:,:]
    #change data labels into categorical. 
    data['n_label'] = data['label'].str.replace("positive", "1")
    data['n_label'] = data['n_label'].str.replace("negative", "0")
    print(data['n_label'])

    # splitting data into training and validation set (70:30)
    xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, data['n_label'], random_state=42, test_size=0.3)
    # Train a Naive Bayes Multinomial classifier
    classifier = MultinomialNB()
    classifier.fit(xtrain_bow, ytrain)
    # making predictions on the testing set 
    y_pred = classifier.predict(xvalid_bow) 

    print("Multinomial Naive Bayes model accuracy(in %):", metrics.accuracy_score(yvalid, y_pred)*100)
    print("Multinomial Naive Bayes model precision(in %):", metrics.precision_score(yvalid, y_pred, pos_label='1')*100)
    print("Multinomial Naive Bayes model F1 score (in %):", metrics.f1_score(yvalid, y_pred, pos_label='1')*100)

    #Classify and count tweet quantifier method
    count = (Counter(y_pred))
    #print(count)
    #pos counts "positive" labels in the y_pred array
    pos = count['positive']
    #neg counts "negative" labels in the y_pred array
    neg = count['negative']
    #print(pos)
    cc = pos/(pos + neg)
    #print(cc)    
    print("The number of positives based on CC approach is "+ str(cc) + ".") 

    #compare yvalid and y_pred to establish the share of true positives and false positives
    #calculate confusion matrix
    confusion_matrix(yvalid, y_pred)
    tn, fp, fn, tp = confusion_matrix(yvalid, y_pred).ravel()
    (tn, fp, fn, tp)
    #calculate tpr and fpr by using the elements of the confusion matrix
    tpr = tp/(tp + fn)
    fpr = fp/(fp + tn)
    #print(tpr, fpr)

    #Quantify using Adjusted Count method
    ac = (cc - fpr)/(tpr - fpr)
    print("The number of positives based on Adjusted Count method is "+ str(ac) + ".") 
    