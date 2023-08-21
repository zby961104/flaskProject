import torch
from rdkit import Chem
import numpy as np
import rdkit

atom_feature_size = 39


def process_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    atom_num = len(mol.GetAtoms())
    mol_atom_feature = np.zeros([atom_num, atom_feature_size])

    adj = np.zeros([atom_num, atom_num])

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atom_feature = get_atom_feature(atom)
        mol_atom_feature[idx] = atom_feature

    for bond in mol.GetBonds():
        start_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        adj[start_atom, end_atom] = 1
        adj[end_atom, start_atom] = 1

    mol_atom_feature = torch.Tensor(mol_atom_feature)
    adj = torch.Tensor(adj)

    return mol_atom_feature, adj

max_atom_num = -1
def process_all_smiles(smiles_list):
    max_atom_num = -1
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        atom_num = len(mol.GetAtoms())
        max_atom_num = max(max_atom_num, atom_num)

    entire_atom_features = []
    entire_adj = []
    entire_atom_mask = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        mol_atom_feature = np.zeros([max_atom_num, atom_feature_size])
        adj = np.zeros([max_atom_num, max_atom_num])
        mol_atom_mask = np.zeros([max_atom_num])

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            mol_atom_mask[idx] = 1.0
            atom_feature = get_atom_feature(atom)
            mol_atom_feature[idx] = atom_feature

        for bond in mol.GetBonds():
            start_atom = bond.GetBeginAtomIdx()
            end_atom = bond.GetEndAtomIdx()
            adj[start_atom, end_atom] = 1
            adj[end_atom, start_atom] = 1

        # mol_atom_feature = torch.Tensor(mol_atom_feature)
        # adj = torch.Tensor(adj)
        # mol_atom_mask = torch.Tensor(mol_atom_mask)

        entire_atom_features.append(mol_atom_feature)
        entire_adj.append(adj)
        entire_atom_mask.append(mol_atom_mask)

    entire_atom_features = torch.Tensor(entire_atom_features)
    entire_adj = torch.Tensor(entire_adj)
    entire_atom_mask = torch.Tensor(entire_atom_mask)

    return entire_atom_features, entire_adj, entire_atom_mask


def get_atom_feature(atom):
    feature = np.zeros(39)

    # Symbol
    symbol = atom.GetSymbol()
    SymbolList = ['B','C','N','O','F','Si','P','S','Cl','As','Se','Br','Te','I','At']
    if symbol in SymbolList:
        loc = SymbolList.index(symbol)
        feature[loc] = 1
    else:
        feature[15] = 1

    # Degree
    degree = atom.GetDegree()
    if degree > 5:
        print("atom degree larger than 5. Please check before featurizing.")
        raise RuntimeError

    feature[16 + degree] = 1

    # Formal Charge
    charge = atom.GetFormalCharge()
    feature[22] = charge

    # radical electrons
    radelc = atom.GetNumRadicalElectrons()
    feature[23] = radelc

    # Hybridization
    hyb = atom.GetHybridization()
    HybridizationList = [rdkit.Chem.rdchem.HybridizationType.SP,
                         rdkit.Chem.rdchem.HybridizationType.SP2,
                         rdkit.Chem.rdchem.HybridizationType.SP3,
                         rdkit.Chem.rdchem.HybridizationType.SP3D,
                         rdkit.Chem.rdchem.HybridizationType.SP3D2]
    if hyb in HybridizationList:
        loc = HybridizationList.index(hyb)
        feature[loc+24] = 1
    else:
        feature[29] = 1

    # aromaticity
    if atom.GetIsAromatic():
        feature[30] = 1

    # hydrogens
    hs = atom.GetNumImplicitHs()
    feature[31+hs] = 1

    # chirality, chirality type
    if atom.HasProp('_ChiralityPossible'):
        feature[36] = 1
        try:
            chi = atom.GetProp('_CIPCode')
            ChiList = ['R','S']
            loc = ChiList.index(chi)
            feature[37+loc] = 1
            #print("Chirality resolving finished.")
        except:
            feature[37] = 0
            feature[38] = 0
    return feature

