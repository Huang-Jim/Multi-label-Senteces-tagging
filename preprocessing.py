#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 13:30:28 2019

@author: jim
"""
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import pdb

#%% Step 1
# Loading the .csv provided from the competetion and drop out {Authors} and {Created Date} columns
train_data_df = pd.read_csv('./task1_trainset.csv')
train_data_df = train_data_df.drop(columns=['Authors', 'Created Date'])
test_data_df = pd.read_csv('./task1_public_testset.csv')
test_data_df = test_data_df.drop(columns=['Authors', 'Created Date'])
categories = np.array(['BACKGROUND', 'OBJECTIVES', 'METHODS', 'RESULTS', 'CONCLUSIONS', 'OTHERS'])
#%% Step 2
# Split abstract into sentences, one row for one sentence.
# Muti-labeling each sentence and save it into [train.csv] and [test.csv]
Sentences = []
for index, row in train_data_df.iterrows():
    label = row['Task 1'].split(' ')
    Abstract = row['Abstract'].split('$$$')
    title = row['Title']
    Id = row['Id']
    for i, v in enumerate(label):
        tmp = [0, 0, 0, 0, 0, 0]
        position = '{}'.format(i+1)
        total_len = '{}'.format(len(Abstract))
        for jj, val in enumerate(v.split('/')):
            tmp[np.where(val==categories)[0][0]] = 1
        n_captal = len(re.findall('([A-Z][a-z]+)', Abstract[i]))
        this = [Id, position, total_len, n_captal, Abstract[i], title]
        this.extend(tmp)
        Sentences.append(this)
df = pd.DataFrame(Sentences, columns = ['Id', 'Position', 'Total_len', 'n_captal', 'Sentences', 'Title', 'BACKGROUND', 'OBJECTIVES', 'METHODS', 'RESULTS', 'CONCLUSIONS', 'OTHERS']) 
df.to_csv('./train.csv', index=None)

Sentences = []
Position = []
for index, row in test_data_df.iterrows():
    Abstract = row['Abstract'].split('$$$')
    title = row['Title']
    Id = row['Id']
    for i, v in enumerate(Abstract):
        tmp = [0, 0, 0, 0, 0, 0]
        position = '{}'.format(i+1)
        total_len = '{}'.format(len(Abstract))
        n_captal = len(re.findall('([A-Z][a-z]+)', Abstract[i]))
        this = [Id, position, total_len, n_captal, Abstract[i], title]
        this.extend(tmp)
        Sentences.append(this)
df = pd.DataFrame(Sentences, columns = ['Id', 'Position', 'Total_len', 'n_captal', 'Sentences', 'Title', 'BACKGROUND', 'OBJECTIVES', 'METHODS', 'RESULTS', 'CONCLUSIONS', 'OTHERS']) 
df.to_csv('./test.csv', index=None)
#%% Step 3
# You can choose or design a new function for preprocessing the sentences.
# Put your own function and apply it in [apply_func]
def sub_text(sen):
    sentence = re.sub('[^a-zA-Z]', ' ', sen)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence
    
stop_words = set(stopwords.words('english'))
def remove_stop(sentence):
    text = sentence.lower().split()
    text = [w for w in text if not w in stop_words and len(w)>=3]
    text = ' '.join(text)
    return text

def cal_words(sentence):
    text = sentence.lower().split()
    return len(text)

stemmer = SnowballStemmer("english")
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

def apply_func(df):
    df['Sentences'] = df['Sentences'].apply(sub_text)
    df['keep_stop'] = df['Sentences'] # I do not remove stop words in this competetion
    df['word_cnt'] = df['Sentences'].apply(cal_words) # count the words in sentences
    #df['Sentences'] = df['Sentences'].apply(remove_stop) # remove stop words
    #df['stemmed_Sentences'] = df['Sentences'].apply(stemming) # do stemming
    return df

print('Cleaning sentences...')
train_data_df = apply_func(pd.read_csv('./train.csv'))
test_data_df = apply_func(pd.read_csv('./test.csv'))
print('Creating preprocess_csv')
train_data_df.to_csv('./after_preprocess_train.csv', index=None)
test_data_df.to_csv('./after_preprocess_test.csv', index=None)
