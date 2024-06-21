from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt


def load_data(max_words=10000, maxlen=200):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)
    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_test = pad_sequences(x_test, maxlen=maxlen)
    return x_train, y_train, x_test, y_test


def create_model(max_words=10000, maxlen=200):
    model = Sequential()
    model.add(Embedding(max_words, 2, input_length=maxlen))
    model.add(SimpleRNN(8))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_test_model(model, x_train, y_train, epochs=15, test=False, x_test=None, y_test=None):
    history = model.fit(x_train,
              y_train,
              epochs=epochs,
              batch_size=128,
              validation_split=0.1)
    model.save('Mood_Ver1.0')
    if test:
        scores = model.evaluate(x_test, y_test, verbose=1)
    else:
        scores = None
    return model, history, scores


def plot(data):
    plt.plot(data.history['acc'],
             label='Доля верных ответов на обучающем наборе')
    plt.plot(data.history['val_acc'],
             label='Доля верных ответов на проверочном наборе')
    plt.xlabel('Эпоха обучения')
    plt.ylabel('Доля верных ответов')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()
    model = create_model()
    model, history, scores = train_test_model(model=model,
                     test=True,
                     x_train=x_train, y_train=y_train,
                     x_test=x_test, y_test=y_test)
    plot(history)


