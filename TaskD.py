
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

data = pd.read_table('SemEval2017-task4-dev.subtask-BD.english.INPUT.txt', index_col=False, header =0, sep='\t', names=['ID','topic','label','tweet'])
print(data.dtypes)
print(data.head(10))

def remove_pattern(input_txt, pattern):
#    print(pattern, input_txt)
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
print(bow)
