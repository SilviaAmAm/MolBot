import numpy as np

from rdkit.Chem import Descriptors, MolFromSmiles
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

def calculate_tpsa_reward(X_strings):
    """
    This function calculates the reward for a list of molecules.
    :param X_strings: SMILES strings
    :type X_string: list of strings
    :return: the valid smiles and the rewards and the index of any invalid smiles
    :rtype: list of float, list of int
    """

    rewards = []
    idx_invalid_smiles = []

    for i, x_string in enumerate(X_strings):
        if len(x_string) == 0:
            idx_invalid_smiles.append(i)
            continue
        m = MolFromSmiles(x_string)

        # If the predicted smiles is invalid, give no reward
        if isinstance(m, type(None)):
            idx_invalid_smiles.append(i)
            continue

        TPSA = Descriptors.TPSA(m)

        # To obtain molecules mostly with polarity between 90 and 120
        rewards.append(np.exp(-(TPSA - 105) ** 2))

    return rewards, idx_invalid_smiles