import collections
import csv
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model, Sequential

DIM_DATASET = 6
SIZE_TRAINING_DATASET = 300
SIZE_TEST_DATASET = 60


def write_csv(path, data):
    with open(path, 'w', newline='') as file:
        my_csv = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        if isinstance(data, collections.Iterable) and isinstance(data[0], collections.Iterable):
            for i in data:
                my_csv.writerow(i)
        else:
            my_csv.writerow(data)


def generation_of_dataset(size):
    dataset = []
    predict = []
    for i in range(size):
        X = np.random.normal(0, 10)
        e = np.random.normal(0, 0.3)
        some_data = []
        some_data.append(np.cos(X) + e)
        some_data.append(-X + e)
        some_data.append(np.sin(X) * X + e)
        some_data.append(X ** 2 + e)
        some_data.append(-np.fabs(X) + 4)
        some_data.append(X - (X ** 2) / 5 + e)
        dataset.append(some_data)
        predict.append([np.sqrt(np.fabs(X)) + e])
    return np.round(np.array(dataset), decimals=3), np.round(np.array(predict), decimals=3)


def create_objects():
    main_input = Input(shape=(DIM_DATASET,), name='main_input')
    encoded = Dense(60, activation='relu')(main_input)
    encoded = Dense(60, activation='relu')(encoded)
    encoded = Dense(6, activation='linear')(encoded)

    input_encoded = Input(shape=(6,), name='input_encoded')
    decoded = Dense(35, activation='relu', kernel_initializer='normal')(input_encoded)
    decoded = Dense(60, activation='relu')(decoded)
    decoded = Dense(60, activation='relu')(decoded)
    decoded = Dense(DIM_DATASET, name="out_aux")(decoded)

    predicted = Dense(64, activation='relu', kernel_initializer='normal')(encoded)
    predicted = Dense(64, activation='relu')(predicted)
    predicted = Dense(64, activation='relu')(predicted)
    predicted = Dense(1, name="out_main")(predicted)

    encoded = Model(main_input, encoded, name="encoder")
    decoded = Model(input_encoded, decoded, name="decoder")
    predicted = Model(main_input, predicted, name="regr")

    return encoded, decoded, predicted, main_input


x_train, y_train = generation_of_dataset(SIZE_TRAINING_DATASET)
x_test, y_test = generation_of_dataset(SIZE_TEST_DATASET)


mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train -= mean
x_train /= std
x_test -= mean
x_test /= std

y_mean = y_train.mean(axis=0)
y_std = y_train.std(axis=0)
y_train -= y_mean
y_train /= y_std
y_test -= y_mean
y_test /= y_std

encoded, decoded, full_model, main_input = create_objects()


keras_model = Sequential()
keras_model.add(Dense(60, activation='relu'))
keras_model.add(Dense(60, activation='relu'))
keras_model.add(Dense(1))

keras_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
keras_model.fit(x_train, y_train, epochs=60, batch_size=5, verbose=1, validation_data=(x_test, y_test))

full_model.compile(optimizer="adam", loss="mse", metrics=['mae'])
full_model.fit(x_train, y_train, epochs=60, batch_size=5, verbose=1, validation_data=(x_test, y_test))

encoded_data = encoded.predict(x_test)
decoded_data = decoded.predict(encoded_data)
regr = full_model.predict(x_test)

write_csv('./x_train.csv', x_train)
write_csv('./y_train.csv', y_train)
write_csv('./x_test.csv', x_test)
write_csv('./y_test.csv', y_test)
write_csv('./encoded.csv', encoded_data)
write_csv('./decoded.csv', decoded_data)
write_csv('./regr.csv', regr)

decoded.save('decoder.h5')
encoded.save('encoder.h5')
full_model.save('full.h5')
