import numpy as np
import joblib
from keras import Sequential
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras.layers import Activation


data = joblib.load("dataset_2.bz")
X = data["X"]
y = data["y"]

idx_to_char = data["idx_to_char"]
char_to_idx = data["char_to_idx"]

n_samples = X.shape[0]
max_size = X.shape[1]
n_feat = X.shape[2]

hidden_neurons_1 = 256
hidden_neurons_2 = 256


model = Sequential()
# This will output (max_size, n_hidden_1)
model.add(LSTM(units=hidden_neurons_1, input_shape=(None, n_feat), return_sequences=True, dropout=0.3))
# This will output (max_size, n_hidden_2)
model.add(LSTM(units=hidden_neurons_2, input_shape=(None, hidden_neurons_1), return_sequences=True, dropout=0.5))
# This will output (max_size, n_feat)
model.add(TimeDistributed(Dense(n_feat), input_shape=(None, hidden_neurons_2)))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

model.fit(X, y, batch_size=10, verbose=1, nb_epoch=3)

X_pred = np.zeros((1, max_size, n_feat))
y_char = ['G']
X_pred[0, 0, char_to_idx['G']] = 1

slice = X_pred[:, :1, :]
print(slice.shape)

for i in range(max_size-1):
    out = model.predict(X_pred[:, :i+1, :])[0][-1]
    idx_out = np.argmax(out)
    X_pred[0, i+1, idx_out] = 1
    if idx_to_char[idx_out] == 'E':
        break
    else:
        y_char.append(idx_to_char[idx_out])

print(y_char)