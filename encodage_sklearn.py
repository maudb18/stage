from rdkit import Chem
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import requests
import numpy as np
import pandas as pd



r = requests.get('https://raw.githubusercontent.com/aspuru-guzik-group/selfies/master/examples/vae_example/datasets/dataJ_250k_rndm_zinc_drugs_clean.txt')


data = r.text.split('\n')[:-1]

elem = ['C', 'N', 'O'#, 'H'
        ]

filtre = []

for S in data:
    mol = Chem.MolFromSmiles(S)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    all_atoms_in_elem = True
    for e in atoms:
        if e not in elem:
            all_atoms_in_elem = False
            break
    if all_atoms_in_elem:
        filtre.append(mol)



df = pd.DataFrame({'smiles': filtre[:5]})
#print(df.head())
df['smiles'] = df['smiles'].apply(Chem.MolToSmiles)
#unique_char = set(df.smiles.apply(list).sum())
unique_char = list(set(df.smiles.apply(list).sum()))
#print(f"All unique characters found in the preprocessed data set:\n{sorted(unique_char)}")


#max_length = len(max(df["smiles"], key=len))
max_length = max(len(Chem.MolToSmiles(smi)) for smi in filtre)


# Function to add padding before one-hot encoding
# after label (integer) encoding
def initial_padding(smiles, max_len):
    """
    Add zeroes to the list of characters
    after integer encoding them

    Parameters
    ----------
    smiles : str
       SMILES string.
    max_len : int
       Maximum length of the SMILES string

    Returns
    -------
    canonical_char_padded : numpy.ndarray
      Canonical character array padded to max_len.
    """
    canonical_char = list(smiles)
    # Perform padding on the list of characters
    canonical_char_padded = np.pad(canonical_char, (0, max_len - len(canonical_char)), "constant")
    return canonical_char_padded

padded_smis=[]

for smi in filtre[:10]:
    res = initial_padding(smiles=Chem.MolToSmiles(smi), max_len=max_length)
    padded_smis.append(res)

#print(len(padded_smis), padded_smis[0])


# Function to add padding after one-hot encoding
def later_padding(ohe_matrix, smiles_maxlen, unique_char):
    """
    Add horizontal and vertical padding
    to the given matrix using numpy.pad() function.

    Parameters
    ----------
    ohe_matrix : ndarray
        Character array.
    smiles_max_len : int
        Maximum length of the SMILES string.
    unique_char : list
        List of unique characters in the string data set.

    Returns
    -------
    padded_matrix : numpy.ndarray
           Padded one-hot encoded matrix of
           shape (unique char in smiles, max smile_length).
    """

    padded_matrix = np.pad(
        ohe_matrix,
        ((0, smiles_maxlen - len(ohe_matrix)), (0, len(unique_char) - len(ohe_matrix[0]))),
        "constant",
    )
    return padded_matrix


# Function to add padding before one-hot encoding
# after label (integer) encoding
def initial_padding(smiles, max_len):
    """
    Add zeroes to the list of characters
    after integer encoding them

    Parameters
    ----------
    smiles : str
       SMILES string.
    max_len : int
       Maximum length of the SMILES string

    Returns
    -------
    canonical_char_padded : numpy.ndarray
      Canonical character array padded to max_len.
    """
    canonical_char = list(smiles)
    # Perform padding on the list of characters
    canonical_char_padded = np.pad(canonical_char, (0, max_len - len(canonical_char)), "constant")
    return canonical_char_padded




# Use Scikit-learn implementation of one-hot encoding
def sklearn_one_hot_encoded_matrix(
        smiles, islaterpadding, isinitialpadding, smiles_maxlen, unique_char
):
    """
    Label and one-hot encodes the SMILES
    using sklearn LabelEncoder and OneHotEncoder implementation.

    Parameters
    ----------
    smiles : str
        SMILES string of a compound.
    islaterpadding : bool
        Paramater is `True` if `later_padding` is required,
        `False` otherwise.
    isinitialpadding : bool
        Paramater is `True` if `initial_padding` is required,
        `False` otherwise.
    smile_maxlen : int
       Maximum length of the SMILES string
    unique_char : list
        List of unique characters in the string data set.

    Returns
    -------
    onehot_encoded : numpy.ndarray
        One-hot encoded matrix of shape
        (chars in individual SMILES, length of individual SMILES).
    """
    # Integer encoding
    canonical_char = list(smiles)
    label_encoder = LabelEncoder()
    # Fit_transform function is used to first fit the data and then transform it
    integer_encoded = label_encoder.fit_transform(canonical_char)

    # If initial padding, add zeros to vector (columns in matrix)
    if isinitialpadding:
        integer_encoded = initial_padding(integer_encoded, smiles_maxlen)

    # One-hot encoding
    onehot_encoder = OneHotEncoder(sparse=False)
    # Reshape the integer encoded data
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    # If later padding, add zeros to ohe matrix
    if islaterpadding:
        onehot_encoded = later_padding(onehot_encoded, smiles_maxlen, unique_char)

    onehot_encoded = onehot_encoded.transpose()

    # If initial padding, add zeros to rows
    if isinitialpadding:
        row_padding = np.zeros(shape=(len(unique_char) - len(onehot_encoded), smiles_maxlen))
        onehot_encoded = np.append(onehot_encoded, row_padding, axis=0)
    return onehot_encoded



onehot_encoded_list = []

for padded_smiles in padded_smis:
    onehot_encoded = sklearn_one_hot_encoded_matrix(
        smiles=padded_smiles,
        islaterpadding=True,
        isinitialpadding=True,
        smiles_maxlen=max_length,
        unique_char=unique_char
    )
    onehot_encoded_list.append(onehot_encoded)

mat = onehot_encoded_list[0]

print(len(onehot_encoded_list), onehot_encoded_list[0])
print(mat.shape)
print(max_length, len(padded_smis))