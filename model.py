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

data = pd.read_table('SemEval2017-task4-dev.subtask-A.english.INPUT.txt', delim_whitespace=False, names=['ID','label','tweet'])
print(data)

def remove_pattern(input_txt, pattern):
    print(pattern, input_txt)
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
  
    return input_txt
#Removing Twitter handle
data['tidy_tweet'] = np.vectorize(remove_pattern)(data['tweet'], "@[\w]*")
#Removing Punctuations, Numbers, and Special Characters
data['tidy_tweet'] = data['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
#Removing short words of 3 letters and less
data['tidy_tweet'] = data['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
#Tokenizing - TO DO: you can also use NLTK library for this
tokenized_tweet = data['tidy_tweet'].apply(lambda x: x.split())

#Stemming 
from nltk.stem.porter import *
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) 

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
data['tidy_tweet'] = tokenized_tweet

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

# splitting data into training and validation set (70:30)
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, data['label'], random_state=42, test_size=0.3)
# Train a Naive Bayes Multinomial classifier
classifier = MultinomialNB()
classifier.fit(xtrain_bow, ytrain)
# making predictions on the testing set 
y_pred = classifier.predict(xvalid_bow) 
#classifier = NaiveBayesClassifier.train(train_bow)
#print("\nAccuracy of the classifier:"), nltk.classify.util.accuracy(classifier, test_bow)
from sklearn import metrics 
print("Multinomial Naive Bayes model accuracy(in %):", metrics.accuracy_score(yvalid, y_pred)*100)

#Below is for NLTK Bayes classifier
#print("\nTop 10 most informative words:")
#for item in classifier.most_informative_features()[:10]:
#    print(item[0])

