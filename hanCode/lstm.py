import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from sklearn import svm
from sklearn.metrics import classification_report
from collections import Counter
import numpy as np
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Conv1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from sklearn.metrics import classification_report
import re
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
# read files
trainData = pd.read_csv("../data/train_data2.csv")
testData = pd.read_csv("../data/test_data3.csv")



REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')


def clean_text(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.lower()  # lowercase text
    text = text.replace('11111111','techA')
    text = text.replace('22222222', 'techB')
    text = REPLACE_BY_SPACE_RE.sub(' ',
                                   text)  # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('',
                              text)  # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing.
    text = text.replace('x', '')
    #    text = re.sub(r'\W+', '', text)
    return text


tranSent = trainData['sentence'].apply(clean_text)
testSent = testData['sentence'].apply(clean_text)



sentences = pd.concat([tranSent, tranSent])
tokenizer = Tokenizer(num_words=50000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(sentences.values)
word_index = tokenizer.word_index

x_train = tokenizer.texts_to_sequences(tranSent.values)
x_train = pad_sequences(x_train, maxlen=80)
print(x_train)
y_train = pd.get_dummies(trainData['polarity']).values
print(y_train)

x_test = tokenizer.texts_to_sequences(testSent.values)
x_test = pad_sequences(x_test, maxlen=80)
y_test = pd.get_dummies(testData['polarity']).values


max_features = 20000
# cut texts after this number of words (among top max_features most common words)
maxlen = 80
batch_size = 32


print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

print('Build model...')
# lstm
# model = Sequential()
# model.add(Embedding(20000, 100, input_length=x_train.shape[1]))
# model.add(SpatialDropout1D(0.2))
# model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(3, activation='softmax'))
#
# # try using different optimizers and different optimizer configs
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

#cnn
model = Sequential()


embedding_layer = Embedding(20000, 100, input_length=maxlen , trainable=False)
model.add(embedding_layer)

model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(3, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

print('Train...')
history = model.fit(x_train, y_train, epochs=15, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

# predicts  = model.predict(x_test, batch_size=batch_size)
re = []
labels=[0,1,2]

for i in testSent:
    new_complaint = [i]
    print(new_complaint)
    seq = tokenizer.texts_to_sequences(new_complaint)
    padded = pad_sequences(seq, maxlen=80)
    print(padded)
    pred = model.predict(padded)
    print(pred)
    re.append(labels[np.argmax(pred)])
print(re)
report = classification_report(testData['polarity'], re, output_dict=True)
print(report)
