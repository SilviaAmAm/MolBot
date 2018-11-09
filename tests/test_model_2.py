from models import sklearn_models as sm
import os
import numpy as np

def test_set_tb():

    try:
        estimator = sm.Model_2(tensorboard=1)
        raise Exception
    except:
        pass

def test_hidden_neurons():

    attempts = [0, 1.5, 'Hello', None]

    for item in attempts:
        try:
            estimator = sm.Model_2(hidden_neurons_1=item)
            raise Exception
        except:
            pass

def test_set_dropout():

    attempts = [-1, 1.5, 'Hello', None]

    for item in attempts:
        try:
            estimator = sm.Model_2(dropout_1=item)
            raise Exception
        except:
            pass

def test_check_smiles():

    attempts = [['ola', 5, 'hello'], [], [None, None], [5, 7, 9]]

    for item in attempts:
        try:
            estimator = sm.Model_2(smiles=item)
            raise Exception
        except:
            pass

def test_initialise_data_fit():
    correct_smiles = ["CC(=O)NC(CS)C(=O)Oc1ccc(NC(C)=O)cc1", "COc1ccc2CC5C3C=CC(O)C4Oc1c2C34CCN5C",
                      "O=C(C)Oc1ccccc1C(=O)O"]
    padded_smiles = ["GCC(=O)NC(CS)C(=O)Oc1ccc(NC(C)=O)cc1E", "GCOc1ccc2CC5C3C=CC(O)C4Oc1c2C34CCN5CE",
                      "GO=C(C)Oc1ccccc1C(=O)OEAAAAAAAAAAAAAA"]
    correct_idx = [0, 1, 2]

    # Case 1: smiles stored in the class
    estimator_1 = sm.Model_2(smiles=correct_smiles)
    X_hot, y_hot = estimator_1._initialise_data_fit(correct_idx)

    # Checking whether X_hot and y_hot correspond to what they should
    decode_X_hot, decode_y_hot = decode_hotness(X_hot, y_hot, estimator_1.idx_to_char)

    # Checking that the decoded smiles match the original padded smiles
    for k, item in enumerate(padded_smiles):
        assert item == decode_X_hot[k]
        assert item[1:] == decode_y_hot[k][:-1]

    del estimator_1

    # Case 2: smiles passed directly to initialise_data_fit
    estimator_2 = sm.Model_2()
    X_hot, y_hot = estimator_2._initialise_data_fit(correct_smiles)

    # Checking whether X_hot and y_hot correspond to what they should
    decode_X_hot, decode_y_hot = decode_hotness(X_hot, y_hot, estimator_2.idx_to_char)

    # Checking that the decoded smiles match the original padded smiles
    for k, item in enumerate(padded_smiles):
        assert item == decode_X_hot[k]
        assert item[1:] == decode_y_hot[k][:-1]

    del estimator_2

def test_initialise_data_predict():

    correct_smiles = ["CC(=O)NC(CS)C(=O)Oc1ccc(NC(C)=O)cc1", "COc1ccc2CC5C3C=CC(O)C4Oc1c2C34CCN5C",
                      "O=C(C)Oc1ccccc1C(=O)O"]
    padded_smiles = ["GCC(=O)NC(CS)C(=O)Oc1ccc(NC(C)=O)cc1E", "GCOc1ccc2CC5C3C=CC(O)C4Oc1c2C34CCN5CE",
                     "GO=C(C)Oc1ccccc1C(=O)OEAAAAAAAAAAAAAA"]
    correct_idx = [0, 1, 2]

    # Case 1: smiles stored in the class
    estimator_1 = sm.Model_2(smiles=correct_smiles)
    X_string, X_hot = estimator_1._initialise_data_predict(correct_idx, frag_length=5)

    # Checking whether X_hot and y_hot correspond to what they should
    decode_X_hot = decode_hotness(X_hot, y_hot=None, idx_to_char=estimator_1.idx_to_char)

    # Checking that the decoded smiles match the original padded smiles
    for k, item in enumerate(padded_smiles):
        assert item[:5] == decode_X_hot[k]
        assert decode_X_hot[k] == X_string[k]

    del estimator_1

    # Case 2: passing smiles directly
    estimator_2 = sm.Model_2()
    _, _ = estimator_2._initialise_data_fit(correct_smiles)
    X_string, X_hot = estimator_2._initialise_data_predict(correct_smiles, frag_length=5)

    # Checking whether X_hot and y_hot correspond to what they should
    decode_X_hot = decode_hotness(X_hot, y_hot=None, idx_to_char=estimator_2.idx_to_char)

    # Checking that the decoded smiles match the original padded smiles
    for k, item in enumerate(padded_smiles):
        assert item[:6] == decode_X_hot[k]  # The 6 is frag_length + 1 to take the 'G' into account
        assert decode_X_hot[k] == X_string[k]

    del estimator_2

def decode_hotness(X_hot, y_hot, idx_to_char):

    with_y_hot = not isinstance(y_hot, type(None))

    # Checking whether X_hot and y_hot correspond to what they should
    decode_X_hot = []
    decode_y_hot = []
    for i in range(len(X_hot)):
        hot_encoded_X = X_hot[i]
        if with_y_hot:
            hot_encoded_y = y_hot[i]

        decoded_smile_X = ""
        decoded_smile_y = ""
        for j in range(hot_encoded_X.shape[0]):
            idx_X = np.argmax(hot_encoded_X[j])
            decoded_smile_X += idx_to_char[idx_X]
            if with_y_hot:
                idx_y = np.argmax(hot_encoded_y[j])
                decoded_smile_y += idx_to_char[idx_y]

        decode_X_hot.append(decoded_smile_X)
        decode_y_hot.append(decoded_smile_y)

    if with_y_hot:
        return decode_X_hot, decode_y_hot
    else:
        return decode_X_hot

def test_fit():

    correct_smiles = ["CC(=O)NC(CS)C(=O)Oc1ccc(NC(C)=O)cc1", "COc1ccc2CC5C3C=CC(O)C4Oc1c2C34CCN5C",
              "O=C(C)Oc1ccccc1C(=O)O"]
    correct_idx = [0, 1, 2]
    incorrect_smiles = [[1, 2, 3], [4, 5, 6]]
    incorrect_idx = [4, 5, 6]

    estimator = sm.Model_2(smiles=correct_smiles)
    estimator.fit(correct_idx)

    estimator = sm.Model_2()
    estimator.fit(correct_smiles)

    try:
        estimator = sm.Model_2(smiles=incorrect_smiles)
        estimator.fit(correct_idx)
        raise Exception
    except:
        pass

    try:
        estimator = sm.Model_2()
        estimator.fit(correct_idx)
        raise Exception
    except:
        pass

    try:
        estimator = sm.Model_2(smiles=correct_smiles)
        estimator.fit(incorrect_idx)
        raise Exception
    except:
        pass

def test_predict():

    correct_smiles = ["CC(=O)NC(CS)C(=O)Oc1ccc(NC(C)=O)cc1", "COc1ccc2CC5C3C=CC(O)C4Oc1c2C34CCN5C",
              "O=C(C)Oc1ccccc1C(=O)O"]
    correct_idx = [0, 1, 2]
    incorrect_smiles = [[1, 2, 3], [4, 5, 6]]
    incorrect_idx = [4, 5, 6]

    estimator = sm.Model_2(smiles=correct_smiles)
    estimator.fit(correct_idx)
    pred_1 = estimator.predict(correct_idx)
    estimator._check_smiles(pred_1)

    estimator = sm.Model_2()
    estimator.fit(correct_smiles)
    pred_2 = estimator.predict(correct_smiles)
    estimator._check_smiles(pred_2)

    estimator = sm.Model_2()
    estimator.fit(correct_smiles)
    pred_3 = estimator.predict()
    estimator._check_smiles(pred_3)

    try:
        estimator = sm.Model_2(smiles=correct_smiles)
        estimator.fit(correct_idx)
        estimator.predict(correct_smiles)
        raise Exception
    except:
        pass

    try:
        estimator = sm.Model_2()
        estimator.fit(correct_smiles)
        estimator.predict(correct_idx)
        raise Exception
    except:
        pass

    try:
        estimator = sm.Model_2(smiles=correct_smiles)
        estimator.fit(correct_idx)
        estimator.predict(incorrect_idx)
        raise Exception
    except:
        pass

def test_score():
    correct_smiles = ["CC(=O)NC(CS)C(=O)Oc1ccc(NC(C)=O)cc1", "COc1ccc2CC5C3C=CC(O)C4Oc1c2C34CCN5C",
                      "O=C(C)Oc1ccccc1C(=O)O"]
    correct_idx = [0, 1, 2]

    estimator = sm.Model_2(smiles=correct_smiles)
    estimator.fit(correct_idx)
    pred_1 = estimator.predict(correct_idx)
    estimator._check_smiles(pred_1)
    try:
        score = estimator.score(correct_idx)
        assert score >= 0
    except ModuleNotFoundError:
        print("Test_score aborted since RDKit is not installed.")

def test_resume():
    correct_smiles = ["CC(=O)NC(CS)C(=O)Oc1ccc(NC(C)=O)cc1", "COc1ccc2CC5C3C=CC(O)C4Oc1c2C34CCN5C",
                      "O=C(C)Oc1ccccc1C(=O)O"]
    estimator = sm.Model_2()
    estimator.fit(correct_smiles)
    estimator.predict()
    estimator.fit(correct_smiles)

def test_save():
    correct_smiles = ["CC(=O)NC(CS)C(=O)Oc1ccc(NC(C)=O)cc1", "COc1ccc2CC5C3C=CC(O)C4Oc1c2C34CCN5C",
                      "O=C(C)Oc1ccccc1C(=O)O"]

    estimator = sm.Model_2(epochs=3)
    estimator.fit(correct_smiles)
    estimator.save(filename="temp")

def test_reload_predict():
    correct_smiles = ["CC(=O)NC(CS)C(=O)Oc1ccc(NC(C)=O)cc1", "COc1ccc2CC5C3C=CC(O)C4Oc1c2C34CCN5C",
                      "O=C(C)Oc1ccccc1C(=O)O"]

    estimator = sm.Model_2(epochs=3)
    estimator.load(filename="temp")
    estimator.predict()
    estimator.predict(correct_smiles, frag_length=5)

def test_rl():
    correct_smiles = ["CC(=O)NC(CS)C(=O)Oc1ccc(NC(C)=O)cc1", "COc1ccc2CC5C3C=CC(O)C4Oc1c2C34CCN5C",
                      "O=C(C)Oc1ccccc1C(=O)O"]

    estimator = sm.Model_2(epochs=3)
    estimator.load(filename="temp")
    estimator.fit_with_rl(temperature=0.75, n_train_episodes=3)
    estimator.predict()
    estimator.predict(correct_smiles, frag_length=5)

def test_reload_fit():
    correct_smiles = ["CC(=O)NC(CS)C(=O)Oc1ccc(NC(C)=O)cc1", "COc1ccc2CC5C3C=CC(O)C4Oc1c2C34CCN5C",
                      "O=C(C)Oc1ccccc1C(=O)O"]

    estimator = sm.Model_2(epochs=3)
    estimator.load(filename="temp")
    estimator.fit(correct_smiles)

    os.remove("temp.h5")
    os.remove("temp.pickle")

if __name__ == "__main__":
    test_set_tb()
    test_hidden_neurons()
    test_set_dropout()
    test_check_smiles()
    test_initialise_data_fit()
    test_initialise_data_predict()
    test_fit()
    test_predict()
    test_score()
    test_resume()
    test_save()
    test_reload_predict()
    test_rl()
    test_reload_fit()