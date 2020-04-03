#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:33:34 2019

@author: jim
"""

import numpy as np
import pandas as pd
import pdb

import tensorflow.keras.backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Lambda,concatenate, Bidirectional, GRU
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score
from keras.utils import plot_model
import joblib

categories = np.array(['BACKGROUND', 'OBJECTIVES', 'METHODS', 'RESULTS', 'CONCLUSIONS', 'OTHERS'])
train_data_df = pd.read_csv('../after_preprocess_train.csv')
test_data_df = pd.read_csv('../after_preprocess_test.csv')
#%% 
# Function for getting the GloVe embedding. I use 50 dims here.
def get_GloVE(vocab_size, tokenizer):
    embeddings_dictionary = dict()
    glove_file = open('../glove.6B/glove.6B.50d.txt', encoding="utf8")
    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    glove_file.close()
    
    embedding_matrix = np.zeros((vocab_size, 50))
    for word, index in tokenizer.word_index.items():
        if word == 'UNK':
            print(word)
        if word == 'OUTOFWORD':
            print(word)
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    return embedding_matrix

#%% Training and validation
train_, val_ = train_test_split(train_data_df, test_size=0.2, shuffle=False)
score, epo = [], []

# get the features from current sentence
maxlen = 64
x_train = train_['keep_stop']
x_val = val_['keep_stop']
# get the features from previous/following sentences
x_train_follow = pd.concat([train_['keep_stop'].iloc[1:], pd.Series([''])], ignore_index=True)
x_train_previous = pd.concat([pd.Series(['']), train_['keep_stop'].iloc[:-1]], ignore_index=True)

x_val_follow = pd.concat([val_['keep_stop'].iloc[1:], pd.Series([''])], ignore_index=True)
x_val_previous = pd.concat([pd.Series(['']), val_['keep_stop'].iloc[:-1]], ignore_index=True)

train_title = train_['Title']
val_title = val_['Title']

tokenizer = Tokenizer(num_words=5000, oov_token='UNK', lower=False)
tokenizer.fit_on_texts(np.concatenate((x_train, train_title)))

train_title = tokenizer.texts_to_sequences(train_title)
val_title = tokenizer.texts_to_sequences(val_title)

x_train = tokenizer.texts_to_sequences(x_train)
x_train_follow = tokenizer.texts_to_sequences(x_train_follow)
x_train_previous = tokenizer.texts_to_sequences(x_train_previous)
y_train = train_[categories].to_numpy()
pos_train = train_['Position'].values.reshape(-1,1)
ttl_train = train_['Total_len'].values.reshape(-1,1)
# meta means the features not related to 'text'
meta_train = np.concatenate((pos_train, ttl_train), axis=1)
meta_train = np.expand_dims(meta_train,axis=1)
meta_train = np.tile(meta_train, (1,maxlen,1))

x_val = tokenizer.texts_to_sequences(x_val)
x_val_follow = tokenizer.texts_to_sequences(x_val_follow)
x_val_previous = tokenizer.texts_to_sequences(x_val_previous)
y_val = val_[categories].to_numpy()
pos_val = val_['Position'].values.reshape(-1,1)
ttl_val = val_['Total_len'].values.reshape(-1,1)

meta_val = np.concatenate((pos_val, ttl_val), axis=1)
meta_val = np.expand_dims(meta_val,axis=1)
meta_val = np.tile(meta_val, (1,maxlen,1))

vocab_size = len(tokenizer.word_index) + 1

x_train = pad_sequences(x_train, padding='post', truncating='pre', maxlen=maxlen)
x_train_follow = pad_sequences(x_train_follow, padding='post', truncating='pre', maxlen=maxlen)
x_train_previous = pad_sequences(x_train_previous, padding='post', truncating='pre', maxlen=maxlen)
x_val = pad_sequences(x_val, padding='post', truncating='pre', maxlen=maxlen)
x_val_follow = pad_sequences(x_val_follow, padding='post', truncating='pre', maxlen=maxlen)
x_val_previous = pad_sequences(x_val_previous, padding='post', truncating='pre', maxlen=maxlen)

val_title = pad_sequences(val_title, padding='post', truncating='pre', maxlen=maxlen)
train_title = pad_sequences(train_title, padding='post', truncating='pre', maxlen=maxlen)

embd_dim = 50 # GloVe embedding dims
embedding_matrix = get_GloVE(vocab_size, tokenizer)

# ↓↓↓↓↓↓↓↓↓ Model construction begin ↓↓↓↓↓↓↓↓↓
now_input = Input(shape=(maxlen,), name='Now_input')
follow_input = Input(shape=(maxlen,), name='Follow_input')
previous_input = Input(shape=(maxlen,), name='Previous_input')
meta_input = Input(shape=(maxlen,2), name='Meta_input')
title_input = Input(shape=(maxlen,), name='Title_input')

embedding_now_sents = Embedding(vocab_size, embd_dim, weights=[embedding_matrix], trainable=True, name='embd_now_setns')(now_input)
embedding_prev_sents = Embedding(vocab_size, embd_dim, weights=[embedding_matrix], trainable=True, name='embd_prev_setns')(previous_input)
embedding_fol_sents = Embedding(vocab_size, embd_dim, weights=[embedding_matrix], trainable=True, name='embd_fol_setns')(follow_input)
embedding_title = Embedding(vocab_size, embd_dim, weights=[embedding_matrix], trainable=True, name='embd_title')(title_input)

permute_embd_now_sents = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)), name='permute_embd_now_sents')(embedding_now_sents)
permute_embd_title = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)), name='permute_embedding_title')(embedding_title)

now_sents_embd = Dense(1, activation='relu', name='now_sents_embd')(permute_embd_now_sents)
title_embd = Dense(1, activation='relu', name='title_embd')(permute_embd_title)

now_sents_embd = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)), name='permute_sents_embd')(now_sents_embd)
title_embd = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)), name='permute_title_embd')(title_embd)

sents_prev_GRU = Bidirectional(GRU(128, dropout=0.5, return_sequences=False), name='BGRU_prev_sents')(embedding_prev_sents)
expand_prev = Lambda(lambda x: K.expand_dims(x, axis=1), name='expand_prev')(sents_prev_GRU)

sents_fol_GRU = Bidirectional(GRU(128, dropout=0.5, return_sequences=False), name='BGRU_fol_sents')(embedding_fol_sents)
expand_fol = Lambda(lambda x: K.expand_dims(x, axis=1), name='expand_fol')(sents_fol_GRU)

MERGE_now_sents_meta = concatenate([embedding_now_sents, meta_input], axis=-1, name='Merge_now_sents_meta')
sents_now_GRU = Bidirectional(GRU(128, dropout=0.5, return_sequences=False), name='BGRU_now_sents')(MERGE_now_sents_meta)
expand_now = Lambda(lambda x: K.expand_dims(x, axis=1), name='expand_now')(sents_now_GRU)

MERGE_all = concatenate([expand_prev, expand_now, expand_fol], axis=1, name='Merge_all')
sents_all_GRU = Bidirectional(GRU(512, dropout=0.5, return_sequences=False), name='BGRU_all')(MERGE_all)

dense_layer_out = Dense(6, activation='sigmoid', name='output')(sents_all_GRU)

model = Model(inputs=[now_input, follow_input, previous_input, meta_input, title_input], outputs=dense_layer_out)
plot_model(model, 'cv_model.png')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

earlystopping = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit([x_train,x_train_follow,x_train_previous,meta_train,train_title], y_train, batch_size=512, epochs=70, callbacks=[earlystopping],
                    verbose=1, validation_data=([x_val,x_val_follow,x_val_previous,meta_val,val_title],y_val))
prediction = model.predict([x_val,x_val_follow,x_val_previous,meta_val,val_title], batch_size=512)
# ↑↑↑↑↑↑↑↑↑ Model construction end ↑↑↑↑↑↑↑↑↑

target = np.zeros((len(val_),6))
for ci, c in enumerate(categories):
    bin_1 = np.where(prediction[:, ci] >= 0.5)
    target[bin_1, ci] = 1
    
score.append(f1_score(y_true=y_val, y_pred=target, average='micro'))
epo.append(history.epoch[-1])
print(score)
print(epo)
print('saving CV prob!!!')
# Save the prediceted probability for ensemble
joblib.dump(prediction, 'CV_prob.pkl')

#%% Testing part

ttt_df = pd.read_csv('../task1_sample_submission.csv')

train_text = train_data_df['keep_stop']
test_text = test_data_df['keep_stop']

x_train_follow = pd.concat([train_data_df['keep_stop'].iloc[1:], pd.Series([''])], ignore_index=True)
x_train_previous = pd.concat([pd.Series(['']), train_data_df['keep_stop'].iloc[:-1]], ignore_index=True)

x_test_follow = pd.concat([test_data_df['keep_stop'].iloc[1:], pd.Series([''])], ignore_index=True)
x_test_previous = pd.concat([pd.Series(['']), test_data_df['keep_stop'].iloc[:-1]], ignore_index=True)

train_title = train_data_df['Title']
test_title = test_data_df['Title']

tokenizer = Tokenizer(num_words=5000, oov_token='UNK')
tokenizer.fit_on_texts(np.concatenate((train_text, train_title)))

train_title = tokenizer.texts_to_sequences(train_title)
test_title = tokenizer.texts_to_sequences(test_title)

x_train = tokenizer.texts_to_sequences(train_text)
x_train_follow = tokenizer.texts_to_sequences(x_train_follow)
x_train_previous = tokenizer.texts_to_sequences(x_train_previous)
y_train = train_data_df[categories].to_numpy()
pos_train = train_data_df['Position'].values.reshape(-1,1)
ttl_train = train_data_df['Total_len'].values.reshape(-1,1)
meta_train = np.concatenate((pos_train, ttl_train), axis=1)
meta_train = np.expand_dims(meta_train,axis=1)
meta_train = np.tile(meta_train, (1,64,1))

x_test = tokenizer.texts_to_sequences(test_text)
x_test_follow = tokenizer.texts_to_sequences(x_test_follow)
x_test_previous = tokenizer.texts_to_sequences(x_test_previous)
y_test = test_data_df[categories].to_numpy()
pos_test = test_data_df['Position'].values.reshape(-1,1)
ttl_test = test_data_df['Total_len'].values.reshape(-1,1)
meta_test = np.concatenate((pos_test, ttl_test), axis=1)
meta_test = np.expand_dims(meta_test,axis=1)
meta_test = np.tile(meta_test, (1,64,1))

vocab_size = len(tokenizer.word_index) + 1

maxlen = 64

x_train = pad_sequences(x_train, padding='post', truncating='pre', maxlen=maxlen)
x_train_follow = pad_sequences(x_train_follow, padding='post', truncating='pre', maxlen=maxlen)
x_train_previous = pad_sequences(x_train_previous, padding='post', truncating='pre', maxlen=maxlen)
x_test = pad_sequences(x_test, padding='post', truncating='pre', maxlen=maxlen)
x_test_follow = pad_sequences(x_test_follow, padding='post', truncating='pre', maxlen=maxlen)
x_test_previous = pad_sequences(x_test_previous, padding='post', truncating='pre', maxlen=maxlen)

test_title = pad_sequences(test_title, padding='post', truncating='pre', maxlen=maxlen)
train_title = pad_sequences(train_title, padding='post', truncating='pre', maxlen=maxlen)

embedding_matrix = get_GloVE(vocab_size, tokenizer)

# ↓↓↓↓↓↓↓↓↓ Model construction begin ↓↓↓↓↓↓↓↓↓
now_input = Input(shape=(maxlen,), name='Now_input')
follow_input = Input(shape=(maxlen,), name='Follow_input')
previous_input = Input(shape=(maxlen,), name='Previous_input')
meta_input = Input(shape=(maxlen,2), name='Meta_input')
title_input = Input(shape=(maxlen,), name='Title_input')

embedding_now_sents = Embedding(vocab_size, embd_dim, weights=[embedding_matrix], trainable=True, name='embd_now_setns')(now_input)
embedding_prev_sents = Embedding(vocab_size, embd_dim, weights=[embedding_matrix], trainable=True, name='embd_prev_setns')(previous_input)
embedding_fol_sents = Embedding(vocab_size, embd_dim, weights=[embedding_matrix], trainable=True, name='embd_fol_setns')(follow_input)
embedding_title = Embedding(vocab_size, embd_dim, weights=[embedding_matrix], trainable=True, name='embd_title')(title_input)

permute_embd_now_sents = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)), name='permute_embd_now_sents')(embedding_now_sents)
permute_embd_title = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)), name='permute_embedding_title')(embedding_title)

now_sents_embd = Dense(1, activation='relu', name='now_sents_embd')(permute_embd_now_sents)
title_embd = Dense(1, activation='relu', name='title_embd')(permute_embd_title)

now_sents_embd = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)), name='permute_sents_embd')(now_sents_embd)
title_embd = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)), name='permute_title_embd')(title_embd)

sents_prev_GRU = Bidirectional(GRU(128, dropout=0.5, return_sequences=False), name='BGRU_prev_sents')(embedding_prev_sents)
expand_prev = Lambda(lambda x: K.expand_dims(x, axis=1), name='expand_prev')(sents_prev_GRU)

sents_fol_GRU = Bidirectional(GRU(128, dropout=0.5, return_sequences=False), name='BGRU_fol_sents')(embedding_fol_sents)
expand_fol = Lambda(lambda x: K.expand_dims(x, axis=1), name='expand_fol')(sents_fol_GRU)

MERGE_now_sents_meta = concatenate([embedding_now_sents, meta_input], axis=-1, name='Merge_now_sents_meta')
sents_now_GRU = Bidirectional(GRU(128, dropout=0.5, return_sequences=False), name='BGRU_now_sents')(MERGE_now_sents_meta)
expand_now = Lambda(lambda x: K.expand_dims(x, axis=1), name='expand_now')(sents_now_GRU)

MERGE_all = concatenate([expand_prev, expand_now, expand_fol], axis=1, name='Merge_all')
sents_all_GRU = Bidirectional(GRU(512, dropout=0.5, return_sequences=False), name='BGRU_all')(MERGE_all)

dense_layer_out = Dense(6, activation='sigmoid', name='output')(sents_all_GRU)
model = Model(inputs=[now_input, follow_input, previous_input, meta_input, title_input], outputs=dense_layer_out)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

plot_model(model, 'train_test_model.png')
history = model.fit([x_train,x_train_follow,x_train_previous,meta_train,train_title], y_train, batch_size=512, epochs=epo[0]+2, verbose=1)
prediction = model.predict([x_test, x_test_follow, x_test_previous, meta_test, test_title], batch_size=512)
# ↑↑↑↑↑↑↑↑↑ Model construction end ↑↑↑↑↑↑↑↑↑

# save the model and predict the testing data
target = np.zeros((len(test_data_df),6))
for c in range(6):
    bin_1 = np.where(prediction[:, c] >= 0.5)
    target[bin_1, c] = 1
ttt_df.loc[0:131166-1, categories.tolist()] = target
print('saveing everything...')
ttt_df.to_csv('./mm.csv', index=None)
joblib.dump(prediction, './mm.pkl')
model.save('mm.h5')
