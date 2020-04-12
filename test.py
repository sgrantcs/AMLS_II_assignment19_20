# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 17:02:53 2020

@author: svetl
"""

import csv
import re
import nltk
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer 
from nltk.tokenize import word_tokenize
from string import punctuation 
from nltk.stem.snowball import SnowballStemmer
from gensim import models, corpora
from nltk.corpus import stopwords
 
#Import libraries: re -Regular Expressions (RegEx) library for parsing strings
#Import libraries: NLTK - Natural Processing Toolkit. 

# Load input data
def load_data(input_file):
    corpus = []
    trainingDataSet = []
    with open(input_file, newline = '') as file:
        reader = csv.reader(file, delimiter = '\t')
        next(reader)
        for line in reader:
            trainingDataSet.append(line[0:3])

    return trainingDataSet

#print(trainingDataSet)

if __name__=='__main__':
    # File containing linewise input data 
    input_file = 'SemEval2017-task4-dev.subtask-A.english.INPUT.txt'

    # Load data
    trainingDataSet = load_data(input_file)
    
    # Create a preprocessor object
class PreProcessTweets:
    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])
    
    #processTweets function loops through all the tweets input, calling processTweet function for every tweet    
    def processTweets(self, list_of_tweets):
        processedTweets=[]
        for tweet in list_of_tweets:
            processedTweets.append(self._processTweet(tweet[0:3]))
        return processedTweets
    
    #processTweet function pre-processes all the text
    #re.sub() specifies a regular expression pattern in the first argument, 
    #a new string in the second argument, 
    #and a string to be processed in the third argument.
    def _processTweet(self, tweet):
        #tweet = tweet.lower() # convert text to lower-case
        tweet = word_tokenize(tweet) # remove repeated characters (helloooooooo into hello)
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # remove URLs, any pattern staring with www.or https:
        tweet = re.sub('@[^\s]+', 'AT_USER', tweet) # remove usernames
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
        return [word for word in tweet if word not in self._stopwords]
 
tweetProcessor = PreProcessTweets()
preprocessedTrainingSet = tweetProcessor.processTweets(trainingDataSet)
#preprocessedTestSet = tweetProcessor.processTweets(testDataSet)

print(preprocessedTrainingSet)