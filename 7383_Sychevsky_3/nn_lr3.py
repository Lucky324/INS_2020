import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plot

def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(train_data.shape)
print(test_data.shape)
print(test_targets)

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

k = 8
num_val_samples = len(train_data) // k
num_epochs = 80
all_scores = []
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                         train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_target = np.concatenate([train_targets[: i * num_val_samples],
                                           train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_target, epochs=num_epochs, batch_size=1,
                        validation_data=(val_data, val_targets))
    mae = history.history['mae']
    v_mae = history.history['val_mae']
    x = range(1, num_epochs + 1)
    all_scores.append(v_mae)
    plot.figure(i + 1)
    plot.plot(x, mae, 'bo', label='Training MAE')
    plot.plot(x, v_mae, 'b', label='Validation MAE')
    plot.title('absolute error')
    plot.ylabel('absolute error')
    plot.xlabel('epochs')
    plot.legend()

average_mae_history = [np.mean([x[i] for x in all_scores]) for i in range(num_epochs)]
plot.figure(0)
plot.plot(range(1, num_epochs + 1), average_mae_history)
plot.xlabel('epochs')
plot.ylabel("mean absolute error")
plot.show()