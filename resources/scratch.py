from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout, SimpleRNN, Embedding
from keras.preprocessing import sequence
import numpy as np
from keras import optimizers
from keras.preprocessing.text import Tokenizer
import xml.etree.ElementTree as ET


tree = ET.parse('./COPA-resources/datasets/copa-dev.xml')
root = tree.getroot()
info_array = []
answer_array_dev = []
ind = 0
for child in root.findall('item'):
    id = child.get('id')
    answer = int(child.get('most-plausible-alternative'))
    relation = child.get('asks-for')
    question = child.find('p').text[:-1]+ ' . '
    alter1 = child.find('a1').text[:-1] + ' . '
    alter2 = child.find('a2').text[:-1] + ' . '
    info1 = (question + alter1 + relation).split()
    info2 = (question + alter2 + relation).split()
    info_array.append(info1)
    info_array.append(info2)
    answer_array_dev.append(answer)
    answer_array_dev.append(answer)
    ind += 1

t = Tokenizer()
t.fit_on_texts(info_array)
input_matrix_dev = sequence.pad_sequences(t.texts_to_sequences(info_array), maxlen=50)

#-------test input
tree = ET.parse('./COPA-resources/datasets/copa-test.xml')
root = tree.getroot()
info_array = []
answer_array_test = []
ind = 0
for child in root.findall('item'):
    id = child.get('id')
    answer = int(child.get('most-plausible-alternative'))
    relation = child.get('asks-for')
    question = child.find('p').text[:-1]+ ' . '
    alter1 = child.find('a1').text[:-1] + ' . '
    alter2 = child.find('a2').text[:-1] + ' . '
    info1 = (question + alter1 + relation).split()
    info2 = (question + alter2 + relation).split()
    info_array.append(info1)
    info_array.append(info2)
    answer_array_test.append(answer)
    answer_array_test.append(answer)
    ind += 1

t.fit_on_texts(info_array)
input_matrix_test = sequence.pad_sequences(t.texts_to_sequences(info_array), maxlen=50)

top_words = len(t.word_index)
def vanilla_rnn(state, lra=0.001, dropout=0.1, num_outputs=1):
    model = Sequential()
    model.add(Embedding(top_words+1, 50, input_length=50))
    model.add(SimpleRNN(units=state, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(num_outputs, activation='sigmoid'))
    adamOpt = optimizers.Adam(learning_rate=lra)
    model.compile(loss='binary_crossentropy', optimizer=adamOpt, metrics=['accuracy'])

    return model


def LSTM_rnn(state, lra=0.001, dropout=0.1, num_outputs=1):
    model = Sequential()
    model.add(Embedding(top_words+1, 50, input_length=50))
    model.add(LSTM(state))
    model.add(Dropout(dropout))
    model.add(Dense(num_outputs, activation='sigmoid'))
    adamOpt = optimizers.Adam(learning_rate=lra)
    model.compile(loss='binary_crossentropy', optimizer=adamOpt, metrics=['accuracy'])

    return model


def run_model(model, X_train, y_train, X_test, y_test, epochs=2, batch_size=128):
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


modelVanilla = vanilla_rnn(100,lra=0.001, dropout=0)
modelLSTM = LSTM_rnn(100)
run_model(modelLSTM,  input_matrix_dev, answer_array_dev, input_matrix_test, answer_array_test, epochs=20, batch_size=64)
