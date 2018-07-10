import numpy as np
import joblib
from keras import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Activation
from keras.callbacks import TensorBoard
from keras.layers import Lambda
from keras.models import load_model

tensorboard = TensorBoard(log_dir='./tb/model_1',
    write_graph=True, write_images=False)
callbacks_list = [tensorboard]

data = joblib.load("dataset_1.bz")
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
# This will output (n_hidden_2,)
model.add(LSTM(units=hidden_neurons_2, input_shape=(None, hidden_neurons_1), return_sequences=False, dropout=0.5))
# This will output (n_feat,)
model.add(Dense(n_feat))
# Modifying the softmax with the `Temperature' parameter
model.add(Lambda(lambda x: x / 1))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

model.fit(X, y, batch_size=500, verbose=1, nb_epoch=4, callbacks=callbacks_list)
model.save("./saved_models/model_1/model_1.h5")

# Predicting smiles
X_pred = X[0]
y_pred = ''

for i in range(X_pred.shape[0]):
    idx = np.argmax(X_pred[i])
    y_pred += idx_to_char[idx]

X_pred = np.reshape(X_pred, (1, X_pred.shape[0], X_pred.shape[1]))

X_pred_temp = X_pred

new_model = load_model("./saved_models/model_1/model_1.h5")

while( y_pred[-1] != 'E'):
    out = new_model.predict(X_pred_temp)
    out_idx = np.argmax(out[0])
    y_pred += idx_to_char[out_idx]
    X_pred_temp[:, :-1, :] = X_pred[:, 1:, :]
    y_pred_hot = np.zeros((1, X_pred.shape[-1]))
    y_pred_hot[:, out_idx] = 1
    X_pred_temp[:, -1, :] = y_pred_hot

    if len(y_pred) == 100:
        print("Disaster!\n")
        break

print(y_pred)






