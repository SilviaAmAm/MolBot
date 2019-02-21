# Copyright (c) Michael Mazanetz (NovaData Solutions LTD.), Silvia Amabilino (NovaData Solutions LTD.,
# University of Bristol), David Glowacki (University of Bristol). All rights reserved.
# Licensed under the GPL. See LICENSE in the project root for license information.

from models import smiles_generator as sg
from models import data_processing, reinforcement_learning
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
    estimator.save(filename="temp.h5")

def test_reload_predict():

    estimator = sg.Smiles_generator(epochs=3)
    estimator.load(filename="temp.h5")
    estimator.predict(X_pred)

def test_reload_fit():

    estimator = sg.Smiles_generator(epochs=3)
    estimator.load(filename="temp.h5")
    estimator.fit(X, y)

    os.remove("temp.h5")

def test_rl():

    try:
        from models import rewards

        current_dir = os.path.dirname(os.path.realpath(__file__))
        model_file = current_dir + "/../data/model.h5"
        data_handler_file = current_dir + "/../data/data_proc.pickle"
        reward_f = rewards.calculate_tpsa_reward
        rl = reinforcement_learning.Reinforcement_learning(model_file=model_file,
                                                           data_handler_file=data_handler_file,
                                                           reward_function=reward_f)
        rl.train(temperature=0.75, epochs=2, n_train_episodes=5, sigma=60)
        rl.save("rl_model.h5")
    except ModuleNotFoundError:
        print("You dont seem to have RDKit installed, so to use Reinforcement Learning you will have to make a new "
              "reward function.")
        pass

def test_after_rl():

    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_handler_file = current_dir + "/../data/data_proc.pickle"

    dh = data_processing.Molecules_processing()
    dh.load(data_handler_file)
    X_pred_rl = dh.get_empty(3)
    estimator = sg.Smiles_generator(epochs=3)
    estimator.load(filename="rl_model.h5")
    estimator.predict(X_pred_rl)

    os.remove("rl_model.h5")

if __name__ == "__main__":
    test_set_tb()
    test_hidden_neurons()
    test_set_dropout()
    test_resume()
    test_save()
    test_reload_fit()
    test_reload_predict()
    test_rl()
    test_after_rl()
