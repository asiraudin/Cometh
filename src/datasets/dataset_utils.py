import pickle
from typing import Any, Sequence
from rdkit import Chem
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


def mol_to_torch_geometric(mol, atom_encoder, smiles):
    adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
    edge_index = adj.nonzero().contiguous().T
    bond_types = adj[edge_index[0], edge_index[1]]
    bond_types[bond_types == 1.5] = 4
    edge_attr = bond_types.long()

    atom_types = []
    for atom in mol.GetAtoms():
        atom_types.append(atom_encoder[atom.GetSymbol()])

    atom_types = torch.Tensor(atom_types).long()

    data = Data(x=atom_types, edge_index=edge_index, edge_attr=edge_attr,
                smiles=smiles)
    return data



def remove_hydrogens(data: Data):
    to_keep = data.x > 0
    new_edge_index, new_edge_attr = subgraph(to_keep, data.edge_index, data.edge_attr, relabel_nodes=True,
                                             num_nodes=len(to_keep))
    return Data(x=data.x[to_keep] - 1,         # Shift onehot encoding to match atom decoder
                edge_index=new_edge_index,
                edge_attr=new_edge_attr)


def save_pickle(array, path):
    with open(path, 'wb') as f:
        pickle.dump(array, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class Statistics:
    def __init__(self, num_nodes, node_types, bond_types, valencies=None):
        self.num_nodes = num_nodes
        print("NUM NODES IN STATISTICS", num_nodes)
        self.node_types = node_types
        self.bond_types = bond_types
        self.valencies = valencies


class RemoveYTransform:
    def __call__(self, data):
        data.y = torch.zeros((1, 0), dtype=torch.float)
        return data


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]
