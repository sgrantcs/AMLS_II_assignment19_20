# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
"""Create Class for pre-processing tweets"""
import re
import nltk
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords 
#Import libraries: re -Regular Expressions (RegEx) library for parsing strings
#Import libraries: NLTK - Natural Processing Toolkit. 

class PreProcessTweets:
    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])
    
    #processTweets function loops through all the tweets input, calling processTweet function for every tweet    
    def processTweets(self, list_of_tweets):
        processedTweets=[]
        for tweet in list_of_tweets:
            processedTweets.append((self._processTweet(tweet["text"]),tweet["label"]))
        return processedTweets
    
    #processTweet function pre-processes all the text
    #re.sub() specifies a regular expression pattern in the first argument, 
    #a new string in the second argument, 
    #and a string to be processed in the third argument.
    def _processTweet(self, tweet):
        tweet = tweet.lower() # convert text to lower-case
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # remove URLs, any pattern staring with www.or https:
        tweet = re.sub('@[^\s]+', 'AT_USER', tweet) # remove usernames
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
        tweet = word_tokenize(tweet) # remove repeated characters (helloooooooo into hello)
        return [word for word in tweet if word not in self._stopwords]


