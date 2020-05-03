
import re
import pandas as pd 
import numpy as np 
from numpy import mean
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
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from collections import Counter
from Processing import process_tweets

def run_task_D():
    # Data preprocessing
    data = pd.read_table('SemEval2017-task4-dev.subtask-BD.english.INPUT.txt', index_col=False, header =0, sep='\t', names=['ID','topic','label','tweet'])
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
    #print(data['n_label'])

    # splitting data into training and validation set (70:30)
    xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, data['n_label'], random_state=42, test_size=0.3)
    # Train a Naive Bayes Multinomial classifier
    classifier = MultinomialNB()
    classifier.fit(xtrain_bow, ytrain)
    # making predictions on the testing set 
    y_pred = classifier.predict(xvalid_bow)
    y_predproba = classifier.predict_proba(xvalid_bow) #outputs probability of two class labels 
    y_predprob1 = classifier.predict_proba(xvalid_bow)[:, 1] #outputs probability of positive class    

    print("Multinomial Task D Naive Bayes model accuracy(in %):", metrics.accuracy_score(yvalid, y_pred)*100)
    print("Multinomial Task D Naive Bayes model precision(in %):", metrics.precision_score(yvalid, y_pred, pos_label='1')*100)
    print("Multinomial Task D Naive Bayes model F1 score (in %):", metrics.f1_score(yvalid, y_pred, pos_label='1')*100)
    print("Multinomial Task D Naive Bayes model Recall score (in %):", metrics.recall_score(yvalid, y_pred, pos_label='1')*100)

    #Classify and count tweet quantifier method
    count = (Counter(y_pred))
    #print(count)
    #pos counts "positive" labels in the y_pred array
    pos = count['1']
    #neg counts "negative" labels in the y_pred array
    neg = count['0']
    #print(pos)
    cc = ((pos/(pos + neg))*100)
    #print(cc)    
    print("The number of positives based on Classify and Count approach (in %) "+ str(cc) + ".") 
    
    #compare yvalid and y_pred to establish the share of true positives and false positives
    #calculate confusion matrix
    confusion_matrix(yvalid, y_pred)
    tn, fp, fn, tp = confusion_matrix(yvalid, y_pred).ravel()
    (tn, fp, fn, tp)
    #calculate tpr and fpr by using the above elements of the confusion matrix
    tpr = tp/(tp + fn)
    fpr = fp/(fp + tn)
    #print(tpr, fpr)

    #Quantify using Adjusted Count method
    ac = ((cc/100 - fpr)/(tpr - fpr))*100
    print("The number of positives based on Adjusted Count method (in %) "+ str(ac) + ".") 
    
    #Quantify using Probabilistic Classify & Count method
    pcc = mean(y_predprob1)*100 #calculate average probability of positive class 1
    print("The share of positive tweets based on Probabilistic Classify & Count method (in %) "+ str(pcc) + ".") 
    