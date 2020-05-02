import matplotlib.pyplot as plt
import numpy as np
from keras import models, layers
from keras.utils import to_categorical
from keras.datasets import imdb
import string


DIMENSIONS = 10000

def vectorize(data, dimension=DIMENSIONS):
    results = np.zeros((len(data), dimension))
    for i, sequence in enumerate(data):
        results[i, sequence] = 1
    return results

def loadData():
    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=DIMENSIONS)
    data = np.concatenate((training_data, testing_data), axis=0)
    targets = np.concatenate((training_targets, testing_targets), axis=0)
    data = vectorize(data, DIMENSIONS)
    targets = np.array(targets).astype("float32")
    return data[10000:], targets[10000:], data[:10000], targets[:10000]

def buildModel():
    model = models.Sequential()
    model.add(layers.Dense(50, activation="relu", input_shape=(DIMENSIONS,)))
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation="linear"))
    model.add(layers.Dropout(0.5, noise_shape=None, seed=None))
    model.add(layers.Dense(100, activation="relu"))
    model.add(layers.Dropout(0.5, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def plots(history):
    plt.title('Training and test accuracy')
    plt.plot(history.history['accuracy'], 'r', label='train')
    plt.plot(history.history['val_accuracy'], 'b', label='test')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

    plt.title('Training and test loss')
    plt.plot(history.history['loss'], 'r', label='train')
    plt.plot(history.history['val_loss'], 'b', label='test')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

def predict(review, model, dimensions=DIMENSIONS):
    punctuation = str.maketrans(dict.fromkeys(string.punctuation))
    review = review.lower().translate(punctuation).split(" ")
    indexes = imdb.get_word_index()
    encoded = []
    for w in review:
        if w in indexes and indexes[w] < dimensions:
            encoded.append(indexes[w])
    review = vectorize([np.array(encoded)], dimensions)
    return model.predict(review)[0][0]

train_x, train_y, test_x, test_y = loadData()
model = buildModel()
history = model.fit(train_x, train_y, epochs=2, batch_size=500, validation_data=(test_x, test_y))
print(model.evaluate(test_x, test_y, verbose=1))
plots(history)

review1 = "It is a fantastic movie! This year's best film!"
review2 = "It is a usual film for one evening."
review3 = "it was very very bad!"
result = predict(review1, model)
print(result)