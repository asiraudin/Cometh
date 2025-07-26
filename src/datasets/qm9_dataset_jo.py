import os
import os.path as osp
import pathlib
from typing import Any, Sequence
import json

import torch
from rdkit import Chem, RDLogger
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from hydra.utils import get_original_cwd

from datasets.dataset_utils import mol_to_torch_geometric, remove_hydrogens, Statistics
from datasets.dataset_utils import load_pickle, save_pickle
from datasets.abstract_dataset import AbstractDatasetInfos, AbstractDataModule
from metrics.metrics_utils import compute_all_statistics
from utils import PlaceHolder


def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


atom_encoder = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
atom_decoder = [key for key in atom_encoder.keys()]


class QM9JoDataset(InMemoryDataset):
    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    processed_url = 'https://data.pyg.org/datasets/qm9_v3.zip'

    def __init__(self, split, root, remove_h: bool, transform=None, pre_transform=None, pre_filter=None):
        self.split = split
        self.remove_h = remove_h

        self.atom_encoder = atom_encoder
        if remove_h:
            self.atom_encoder = {k: v - 1 for k, v in self.atom_encoder.items() if k != 'H'}

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

        self.statistics = Statistics(num_nodes=load_pickle(self.processed_paths[1]),
                                     node_types=torch.from_numpy(np.load(self.processed_paths[2])).float(),
                                     bond_types=torch.from_numpy(np.load(self.processed_paths[3])).float(),
                                     valencies=load_pickle(self.processed_paths[4]))
        self.smiles = load_pickle(self.processed_paths[5])

    @property
    def raw_file_names(self):
        return ['qm9.csv', 'valid_idx_qm9.json']

    @property
    def processed_file_names(self):
        h = 'noh' if self.remove_h else 'h'
        if self.split == 'train':
            return [f'train_{h}.pt', f'train_n_{h}.pickle', f'train_atom_types_{h}.npy', f'train_bond_types_{h}.npy',
                    f'train_valency_{h}.pickle', 'train_smiles.pickle']
        else:
            return [f'test_{h}.pt', f'test_n_{h}.pickle', f'test_atom_types_{h}.npy', f'test_bond_types_{h}.npy',
                    f'test_valency_{h}.pickle', 'test_smiles.pickle']

    def download(self):
        pass

    def process(self):
        RDLogger.DisableLog('rdApp.*')
        smiles_list = pd.read_csv(self.raw_paths[0], index_col=0)['SMILES1'].values
        with open(self.raw_paths[1]) as f:
            test_idx = json.load(f)

        test_idx = test_idx['valid_idxs']
        test_idx = [int(i) for i in test_idx]

        train_idx = [i for i in range(len(smiles_list)) if i not in test_idx]

        idxs = train_idx if self.split == 'train' else test_idx

        data_list = []
        smiles_kept = []
        for i, smile in enumerate(tqdm(smiles_list)):
            if i not in idxs:
                continue
            mol = Chem.MolFromSmiles(smile)
            if not self.remove_h:
                mol = Chem.AddHs(mol)
            Chem.Kekulize(mol)
            data = mol_to_torch_geometric(mol, atom_encoder, smile)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        statistics = compute_all_statistics(data_list, self.atom_encoder)

        save_pickle(statistics.num_nodes, self.processed_paths[1])
        np.save(self.processed_paths[2], statistics.node_types)
        np.save(self.processed_paths[3], statistics.bond_types)
        save_pickle(statistics.valencies, self.processed_paths[4])
        save_pickle(set(smiles_list), self.processed_paths[5])
        torch.save(self.collate(data_list), self.processed_paths[0])


class QM9JoDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(get_original_cwd()).parents[0]
        root_path = os.path.join(base_path, self.datadir)

        train_dataset = QM9JoDataset(split='train', root=root_path, remove_h=cfg.dataset.remove_h)
        test_dataset = QM9JoDataset(split='test', root=root_path, remove_h=cfg.dataset.remove_h)

        self.statistics = {'train': train_dataset.statistics,
                           'val': test_dataset.statistics,
                           'test': test_dataset.statistics}
        self.remove_h = cfg.dataset.remove_h
        super().__init__(cfg, train_dataset=train_dataset, val_dataset=test_dataset, test_dataset=test_dataset)


class QM9infos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg):
        self.name = 'qm9'
        self.is_molecular = True
        self.remove_h = cfg.dataset.remove_h

        self.statistics = datamodule.statistics
        self.atom_encoder = atom_encoder
        self.atom_decoder = atom_decoder
        if self.remove_h:
            self.atom_encoder = {k: v - 1 for k, v in self.atom_encoder.items() if k != 'H'}
            self.atom_decoder = [key for key in self.atom_encoder.keys()]
        super().complete_infos(datamodule.statistics)

        self.train_smiles = datamodule.train_dataset.smiles
        self.test_smiles = datamodule.test_dataset.smiles

        self.input_dims = PlaceHolder(X=self.num_node_types, E=5, y=1)
        self.output_dims = PlaceHolder(X=self.num_node_types, E=5, y=0)
        self.max_weight = 390
        self.atom_weights = {1: 12, 2: 14, 3: 16, 4: 19} if self.remove_h else {0: 1, 1: 12, 2: 14, 3: 16, 4: 19}
        self.valencies = [4, 3, 2, 1] if self.remove_h else [1, 4, 3, 2, 1]


# def analyze_dataset(dataset_infos, train_dataloader, train_smiles):
#     all_molecules = []
#     for i, data in enumerate(train_dataloader):
#         dense_data = to_dense(data, dataset_infos.num_atom_types, dataset_infos.num_edge_types)
#         dense_data = dense_data.collapse()
#         X, E = dense_data.X, dense_data.E
#
#         for k in range(X.size(0)):
#             n = int(torch.sum((X != -1)[k, :]))
#             atom_types = X[k, :n].cpu()
#             edge_types = E[k, :n, :n].cpu()
#             all_molecules.append(Molecule(atom_types=atom_types,
#                                           bond_types=edge_types,
#                                           atom_decoder=dataset_infos.atom_decoder))
#
#     print("Evaluating the dataset -- number of molecules to evaluate", len(all_molecules))
#     metrics = SamplingMetrics(train_smiles=train_smiles, dataset_infos=dataset_infos, test=False)
#     metrics(all_molecules, "qm9_analysis", 0, 0)

