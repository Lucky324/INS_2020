from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow_core.python.keras.layers import Dropout

from var2 import gen_data

epochs = 7
img_size=50

def loadData():
    data, labels = gen_data()
    data = data.reshape(data.shape[0], img_size, img_size, 1)
    encoder = LabelEncoder()
    encoder.fit(labels.ravel())
    labels = encoder.transform(labels.ravel())
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)
    return train_data, test_data, train_labels, test_labels

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(50, 50, 1)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

train_data, test_data, train_labels, test_labels = loadData()
model.fit(train_data, train_labels,  epochs=epochs, batch_size=10, validation_data=(test_data, test_labels))
accuracy = model.evaluate(test_data, test_labels)
print("Model accuracy: %s" % (accuracy[1]))