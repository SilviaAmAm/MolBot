from models import data_processing
import os

def _get_data():
    # Reading the data set
    current_dir = os.path.dirname(os.path.realpath(__file__))
    in_d = open(current_dir + "/../data/bioactivity_PPARg_filtered.csv", 'r')

    molecules = []

    for line in in_d:
        line_split = line.split(",")
        molecule_raw = line_split[-3]
        molecule = molecule_raw[1:-1]
        if molecule == "CANONICAL_SMILES":
            pass
        else:
            molecules.append(molecule)
    return molecules

def test_onehot_encode():

    molecules = _get_data()

    data_handler = data_processing.Molecules_processing()
    hot_mols = data_handler.onehot_encode(molecules)
    mols = data_handler.onehot_decode(hot_mols)

    for i in range(len(molecules)):
        assert molecules[i] == mols[i]

if __name__ == "__main__":
    test_onehot_encode()