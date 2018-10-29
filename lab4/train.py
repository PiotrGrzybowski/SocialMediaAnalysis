import pickle
import re
import string

# Others
import nltk
import numpy as np
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
# Keras
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords

## Plotly
# py.init_notebook_mode(connected=True)
from lab4.utils import clean_text

with open('raw.pickle', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    p = u.load()
    print(p.keys())


embeddings_index = dict()
f = open('glove.6B.300d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


labels = np.array([row['label'] for row in p['info']])
texts = p['texts']
texts = [clean_text(text) for text in texts]

unique_words = len(set(" ".join(texts).split()))
longest_sentence = max([len(text.split()) for text in texts])

vocabulary_size = 20000
tokenizer = Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
data = pad_sequences(sequences, maxlen=80)

embedding_matrix = np.zeros((vocabulary_size, 300))
for word, index in tokenizer.word_index.items():
    if index > vocabulary_size - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector


# def create_conv_model():
#     model_conv = Sequential()
#     model_conv.add(Embedding(20000, 300, input_length=80))
#     model_conv.add(Dropout(0.2))
#     model_conv.add(Conv1D(64, 5, activation='relu'))
#     model_conv.add(MaxPooling1D(pool_size=4))
#     model_conv.add(LSTM(300))
#     model_conv.add(Dense(7, activation='softmax'))
#     model_conv.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model_conv
#
#
# model = create_conv_model()

model_glove = Sequential()
model_glove.add(Embedding(vocabulary_size, 300, input_length=80, weights=[embedding_matrix], trainable=False))
model_glove.add(Dropout(0.2))
model_glove.add(Conv1D(64, 5, activation='relu'))
model_glove.add(MaxPooling1D(pool_size=4))
model_glove.add(LSTM(300))
model_glove.add(Dense(7, activation='softmax'))
model_glove.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_glove.fit(data, labels, validation_split=0.2, epochs=3)
model_glove.save('my_model.h5')
