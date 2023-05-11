from rdkit import Chem
from sklearn.preprocessing import OneHotEncoder
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



'''
for n in filtre:
    n = Chem.MolToSmiles(n)


print(len(filtre))
for n in filtre[:5]:
    print(Chem.MolToSmiles(n))
'''

df = pd.DataFrame({'smiles': filtre[:5]})
#print(df.head())
df['smiles'] = df['smiles'].apply(Chem.MolToSmiles)
#unique_char = set(df.smiles.apply(list).sum())
unique_char = list(set(df.smiles.apply(list).sum()))
#print(f"All unique characters found in the preprocessed data set:\n{sorted(unique_char)}")


max_length = len(max(df["smiles"], key=len))
#print(max_length)

def to_list(smiles):
    return list(smiles)
df['smiles_list'] = df['smiles'].apply(to_list)

#print(df)

# Ajouter du padding aux séquences de caractères
padded_sequences = []
for sequence in df['smiles_list']:
    pad_size = max_length - len(sequence)
    padded_sequence = np.pad(sequence, (0, pad_size), 'constant', constant_values=0)
    padded_sequences.append(padded_sequence)

# Convertir la matrice numpy en DataFrame
padded_df = pd.DataFrame({'padded_smiles': padded_sequences})

#print(padded_df.shape)

padded_sequences = np.vstack(padded_df['padded_smiles'].values)


# Créer un objet OneHotEncoder
encoder = OneHotEncoder(sparse=False)

encoded_sequences = encoder.fit_transform(padded_sequences.reshape(-1, 1))
print(encoded_sequences)
