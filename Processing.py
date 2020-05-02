#Required libraries to run this function:
import re
import numpy as np
import string
import nltk
from nltk.stem.porter import *

def remove_pattern(input_txt, pattern):
#    print(pattern, input_txt)
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
  
    return input_txt
        
def process_tweets(data):
    '''
    returns a cleaned up and tokenized "tidy_tweet"
    *data*: pandas frame, "tweet": label
    '''
    #Removing Twitter handle
    data['tidy_tweet'] = np.vectorize(remove_pattern)(data['tweet'], "@[\w]*")
    #Removing Punctuations, Numbers, and Special Characters
    data['tidy_tweet'] = data['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
    #Removing short words of 3 letters and less
    data['tidy_tweet'] = data['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
    #Tokenizing - TO DO: you can also use NLTK library for this
    tokenized_tweet = data['tidy_tweet'].apply(lambda x: x.split())
    #Stemming using Porter Stemmer
    stemmer = PorterStemmer()
    tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) 
    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
    data['tidy_tweet'] = tokenized_tweet

    return data
