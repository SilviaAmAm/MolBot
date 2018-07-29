import sklearn_models as sm

def test_set_tb():

    try:
        estimator = sm.Model_1(tensorboard=1)
        raise Exception
    except:
        pass

def test_hidden_neurons():

    attempts = [0, 1.5, 'Hello', None]

    for item in attempts:
        try:
            estimator = sm.Model_1(hidden_neurons_1=item)
            raise Exception
        except:
            pass

def test_set_dropout():

    attempts = [-1, 1.5, 'Hello', None]

    for item in attempts:
        try:
            estimator = sm.Model_1(dropout_1=item)
            raise Exception
        except:
            pass

def test_check_smiles():

    attempts = [['ola', 5, 'hello'], [], [None, None], [5, 7, 9]]

    for item in attempts:
        try:
            estimator = sm.Model_1(smiles=item)
            raise Exception
        except:
            pass

def test_fit():

    correct_smiles = ["CC(=O)NC(CS)C(=O)Oc1ccc(NC(C)=O)cc1", "COc1ccc2CC5C3C=CC(O)C4Oc1c2C34CCN5C",
              "O=C(C)Oc1ccccc1C(=O)O"]
    correct_idx = [0, 1, 2]
    incorrect_smiles = [[1, 2, 3], [4, 5, 6]]
    incorrect_idx = [4, 5, 6]

    estimator = sm.Model_1(smiles=correct_smiles)
    estimator.fit(correct_idx)

    estimator = sm.Model_1()
    estimator.fit(correct_smiles)

    try:
        estimator = sm.Model_1(smiles=incorrect_smiles)
        estimator.fit(correct_idx)
        raise Exception
    except:
        pass

    try:
        estimator = sm.Model_1()
        estimator.fit(correct_idx)
        raise Exception
    except:
        pass

    try:
        estimator = sm.Model_1(smiles=correct_smiles)
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

    estimator = sm.Model_1(smiles=correct_smiles)
    estimator.fit(correct_idx)
    pred_1 = estimator.predict(correct_idx)
    estimator._check_smiles(pred_1)

    estimator = sm.Model_1()
    estimator.fit(correct_smiles)
    pred_2 = estimator.predict(correct_smiles)
    estimator._check_smiles(pred_2)

    try:
        estimator = sm.Model_1(smiles=correct_smiles)
        estimator.fit(correct_idx)
        estimator.predict(correct_smiles)
        raise Exception
    except:
        pass

    try:
        estimator = sm.Model_1()
        estimator.fit(correct_smiles)
        estimator.predict(correct_idx)
        raise Exception
    except:
        pass

    try:
        estimator = sm.Model_1(smiles=correct_smiles)
        estimator.fit(correct_idx)
        estimator.predict(incorrect_idx)
        raise Exception
    except:
        pass

def test_score():
    correct_smiles = ["CC(=O)NC(CS)C(=O)Oc1ccc(NC(C)=O)cc1", "COc1ccc2CC5C3C=CC(O)C4Oc1c2C34CCN5C",
                      "O=C(C)Oc1ccccc1C(=O)O"]
    correct_idx = [0, 1, 2]
    incorrect_smiles = [[1, 2, 3], [4, 5, 6]]
    incorrect_idx = [4, 5, 6]

    estimator = sm.Model_1(smiles=correct_smiles)
    estimator.fit(correct_idx)
    pred_1 = estimator.predict(correct_idx)
    estimator._check_smiles(pred_1)
    score = estimator.score(correct_idx)
    assert score >= 0

    estimator = sm.Model_1()
    estimator.fit(correct_smiles)
    pred_2 = estimator.predict(correct_smiles)
    estimator._check_smiles(pred_2)
    score = estimator.score(correct_smiles)
    assert score >= 0


if __name__ == "__main__":
    test_set_tb()
    test_hidden_neurons()
    test_set_dropout()
    test_check_smiles()
    test_fit()
    test_predict()
    test_score()