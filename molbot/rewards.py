# Copyright (c) Michael Mazanetz (NovaData Solutions LTD.), Silvia Amabilino (NovaData Solutions LTD.,
# University of Bristol), David Glowacki (University of Bristol). All rights reserved.
# Licensed under the GPL. See LICENSE in the project root for license information.

"""
This module contains rewards functions that can be used in the reinforcement learning.
"""

import numpy as np

from rdkit.Chem import Descriptors, MolFromSmiles
from rdkit.Chem.AllChem import  GetMorganFingerprintAsBitVect
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

def calculate_tpsa_reward(X_strings):
    """
    This function calculates the reward for a list of molecules.

    :param X_strings: SMILES strings
    :type X_string: list of strings
    :return: the rewards (between -1 and 1)
    :rtype: list of float
    """

    rewards = []

    for i, x_string in enumerate(X_strings):
        if len(x_string) == 0:
            rewards.append(-1)
            continue
        m = MolFromSmiles(x_string)

        # If the predicted smiles is invalid, give no reward
        if isinstance(m, type(None)):
            rewards.append(-1)
            continue

        TPSA = Descriptors.TPSA(m)

        # To obtain molecules mostly with polarity between 90 and 120
        rewards.append(np.exp(-(TPSA - 105) ** 2))

    return rewards

def calculate_pic50_reward(X_strings, model="./model.pickle"):
    """
    This function calculates the reward for a list of molecules.

    :param X_strings: SMILES strings
    :type X_string: list of strings
    :param model: path to the model to use to calculate the activities
    :type model: string
    :return: the rewards
    :rtype: list of float
    """

    # Locate the model to use to calculate activities
    try:
        import pickle

        predictor = pickle.load(open(model, "rb"))

    except ModuleNotFoundError:
        print("This function requires the module pickle to unpickle a saved model.")
        exit()

    rewards = []

    for i, x_string in enumerate(X_strings):
        if len(x_string) == 0:
            rewards.append(-1)
            continue

        m = MolFromSmiles(x_string)

        # If the predicted smiles is invalid, give no reward
        if isinstance(m, type(None)):
            rewards.append(-1)
            continue

        # Turning the SMILE in Morgan Fingerprint
        fp_string = GetMorganFingerprintAsBitVect(m, radius=3, nBits=2048)

        pic50 = predictor.predict([fp_string])[0]

        # To obtain molecules mostly with pIC50 larger than 9
        rewards.append(np.tanh(pic50 - 7))

    return rewards
