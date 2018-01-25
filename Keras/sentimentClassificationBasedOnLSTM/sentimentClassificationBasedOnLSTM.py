# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 20:05:00 2017

@author: jercas
"""
from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import nltk
import collections
import numpy as np

#nltk.download('punkt')
# max length of sentence
maxlen = 0 
# word frequency
word_freqs = collections.Counter()
# num of examples/records
num_recs = 0
"""
Train data: 7086 lines.
Label: 1 (positive sentiment)|0 (negative sentiment) sentence.

Test data: 33052 lines, each contains one sentence. 
Unlabeled.
"""
with open('./training.txt','r+') as fp:
    for line in fp:
        label, sentence = line.strip().split("\t")
        words = nltk.word_tokenize(sentence.lower())
        # get maxlen
        if len(words) > maxlen:
            maxlen = len(words)
        # get word frequency
        for word in words:
            word_freqs[word] += 1
        # accumulate num of examples/records
        num_recs += 1
print('max_len ', maxlen)
print('nb_words ', len(word_freqs))