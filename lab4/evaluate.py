import numpy as np
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

from lab4.utils import clean_text

model = load_model('my_model.h5')

texts = ['It was my fault I let you control me',
         'If I was gone tomorrow would you regret what you did to me today',
         'I blame myself for everything that happens',
         'Ahh yes....good morning @davidlabrava I hope you have a good day #Happy',
         'tell him much of their fans are infected with their own terrible disease..... cruel untempered #hate ',
         'There are some at Trump rallies who love his hate speeches against the media, immigrants and critics of any kind. They love to hate',
         'A Farce FRAUD and Waste of resources and money! This is @realDonaldTrump using #Hate to divide as well as desperate move to stay in power! #VoteBlueToSaveAmerica from the Nationalist #Republicans']

texts = [clean_text(text) for text in texts]

unique_words = len(set(" ".join(texts).split()))
longest_sentence = max([len(text.split()) for text in texts])

vocabulary_size = 20000
tokenizer = Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
data = pad_sequences(sequences, maxlen=80)

result = model.predict(data)

types = ['joy', 'fear', 'anger', 'sadness', 'disgust', 'shame', 'guilt']

k = [np.argmax(res) for res in result]
print(type(k))
print([types[i] for i in k])

#'sadness', 'joy', 'guilt', 'joy', 'joy', 'sadness', 'sadness'