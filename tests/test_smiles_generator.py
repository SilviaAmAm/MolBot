from models import smiles_generator as sg
from models import data_processing
import os
import numpy as np

# Data for the tests
smiles = ["CC(=O)NC(CS)C(=O)Oc1ccc(NC(C)=O)cc1", "COc1ccc2CC5C3C=CC(O)C4Oc1c2C34CCN5C",
              "O=C(C)Oc1ccccc1C(=O)O"]
dp = data_processing.Molecules_processing()
X = dp.onehot_encode(smiles)
y = np.zeros(X.shape)
y[:, :-1, :] = X[:, 1:, :]
X_pred = dp.get_empty(3)

def test_set_tb():

    try:
        estimator = sg.Smiles_generator(tensorboard=1)
        raise Exception
    except:
        pass

def test_hidden_neurons():

    attempts = [0, 1.5, 'Hello', None]

    for item in attempts:
        try:
            estimator = sg.Smiles_generator(hidden_neurons_1=item)
            raise Exception
        except:
            pass

def test_set_dropout():

    attempts = [-1, 1.5, 'Hello', None]

    for item in attempts:
        try:
            estimator = sg.Smiles_generator(dropout_1=item)
            raise Exception
        except:
            pass

def test_model():

    estimator = sg.Smiles_generator()
    estimator.fit(X, y)
    estimator.predict(X_pred)

def test_resume():

    estimator = sg.Smiles_generator()
    estimator.fit(X, y)
    estimator.predict(X_pred)
    estimator.fit(X, y)

def test_save():

    estimator = sg.Smiles_generator(epochs=6)
    estimator.fit(X, y)
    estimator.save(filename="temp")

def test_reload_predict():

    estimator = sg.Smiles_generator(epochs=3)
    estimator.load(filename="temp")
    estimator.predict(X_pred)

def test_rl():

    estimator = sg.Smiles_generator(epochs=3)
    estimator.load(filename="temp")
    estimator.fit_with_rl(n_train_episodes=3, data_handler=dp)
    estimator.predict(X_pred)

def test_reload_fit():

    estimator = sg.Smiles_generator(epochs=3)
    estimator.load(filename="temp")
    estimator.fit(X, y)

    os.remove("temp.h5")

if __name__ == "__main__":
    # test_set_tb()
    # test_hidden_neurons()
    # test_set_dropout()
    # test_resume()
    # test_save()
    # test_reload_predict()
    test_rl()
    test_reload_fit()