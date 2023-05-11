import numpy as np
from rdkit import Chem
import requests


#récupération des chaînes de caractères (smiles format str) dans la liste "data"
r = requests.get('https://raw.githubusercontent.com/aspuru-guzik-group/selfies/master/examples/vae_example/datasets/dataJ_250k_rndm_zinc_drugs_clean.txt')
data = r.text.split('\n')[:-1]

#initialisation des éléments que l'on va rechercher
elem = ['C', 'N', 'O'#, 'H'
        ]

#initialisation de la liste "filtre" dans laquelle on va ajouter les molécules qui répondent à la condition
filtre = []

#itération sur les molécules de la liste "data"
#pour ne récupérer que celles qui contiennent tous les éléments de la liste elem
#et uniquement ceux-là
for S in data:
    mol = Chem.MolFromSmiles(S)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    all_atoms_in_elem = True
    for e in atoms:
        if e not in elem:
            all_atoms_in_elem = False
            break
    if all_atoms_in_elem:
        filtre.append(Chem.MolToSmiles(mol))


def smile_to_hot(smile, largest_smile_len, alphabet):
    """
    Convert a single smile string to a one-hot encoding.
    """
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))

    # pad with ' '
    smile = smile.lower() + ' ' * (largest_smile_len - len(smile))

    # integer encode input smile
    integer_encoded = [char_to_int[char] for char in smile]

    # one hot-encode input smile
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)
    return integer_encoded, np.array(onehot_encoded)


def multiple_smile_to_hot(smiles_list, largest_molecule_len, alphabet):
    """
    Convert a list of smile strings to a one-hot encoding

    Returned shape (num_smiles x len_of_largest_smile x len_smile_encoding)
    """
    hot_list = []
    for smile in smiles_list:
        _, onehot_encoded = smile_to_hot(smile, largest_molecule_len, alphabet)
        hot_list.append(onehot_encoded)
    return np.array(hot_list)


max_len = max(len(s) for s in filtre)
result = multiple_smile_to_hot(smiles_list=filtre, largest_molecule_len=max_len, alphabet="#()-/\\12345678 =+@chno[]")
print(result)