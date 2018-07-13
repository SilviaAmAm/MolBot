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

data_1 = joblib.load("dataset_1.bz")
X_1 = data_1["X"]
y_1 = data_1["y"]

idx_to_char_1 = data_1["idx_to_char"]
char_to_idx_1 = data_1["char_to_idx"]

n_samples = X_1.shape[0]
max_size = X_1.shape[1]
n_feat = X_1.shape[2]

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

model.fit(X_1, y_1, batch_size=500, verbose=1, nb_epoch=4, callbacks=callbacks_list)
model.save("./saved_models/model_1/model_1.h5")

new_model = load_model("./saved_models/model_1/model_1.h5")
f = open("./pred_smiles/pred_smiles_1.txt", 'w')

data_2 = joblib.load("dataset_2.bz")
X_2 = data_2["X"]
idx_to_char_2 = data_2["idx_to_char"]
char_to_idx_2 = data_2["char_to_idx"]


for i in range(4):
    # Predicting smiles
    X_original = X_2[i]
    y_orig = ''
    X_pred = X_2[i, :10, :]
    y_pred = ''

    for i in range(X_pred.shape[0]):
        idx = np.argmax(X_pred[i])
        y_pred += idx_to_char_2[idx]

    for i in range(X_original.shape[0]):
        idx = np.argmax(X_original[i])
        y_orig += idx_to_char_2[idx]

    X_pred = np.reshape(X_pred, (1, X_pred.shape[0], X_pred.shape[1]))
    X_pred_temp = X_pred

    while( y_pred[-1] != 'E'):
        out = new_model.predict(X_pred_temp)
        out_idx = np.argmax(out[0])
        y_pred += idx_to_char_1[out_idx]
        X_pred_temp[:, :-1, :] = X_pred[:, 1:, :]
        y_pred_hot = np.zeros((1, X_pred.shape[-1]))
        y_pred_hot[:, out_idx] = 1
        X_pred_temp[:, -1, :] = y_pred_hot

        if len(y_pred) == 100:
            print("Disaster!\n")
            break

    f.write(y_orig + "\t" + y_pred)
    f.write("\n")






