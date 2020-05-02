import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
#import seaborn as sns
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.classify import NaiveBayesClassifier
#import Processing as pr
from Processing import process_tweets
from TaskA import run_task_A
from TaskD import run_task_D

# ======================================================================================================================
# Data preprocessing - can be run to show the results of clean processed tweets
#data = pd.read_table('SemEval2017-task4-dev.subtask-A.english.INPUT.txt', index_col=False, header =0, sep='\t', names=['ID','label','tweet'])
#process_tweets(data)
#print(data.head(10))
# ======================================================================================================================
# Task A - Sentiment Analysis in Twitter

run_task_A()

# ======================================================================================================================
# Task D
# Data preprocessing- can be run to show the results of clean processed tweets
#data = pd.read_table('SemEval2017-task4-dev.subtask-BD.english.INPUT.txt', index_col=False, header =0, sep='\t', names=['ID','topic','label','tweet'])
#process_tweets(data)
#print(data.head(10))

run_task_D()

# ======================================================================================================================
