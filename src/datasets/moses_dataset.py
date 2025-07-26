import os
import os.path as osp
import pathlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from rdkit import Chem, RDLogger
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch_geometric.data import InMemoryDataset, download_url
from hydra.utils import get_original_cwd

from datasets.dataset_utils import mol_to_torch_geometric, Statistics, load_pickle, save_pickle, to_list
from datasets.abstract_dataset import AbstractDatasetInfos, AbstractDataModule
from metrics.metrics_utils import compute_all_statistics
from utils import PlaceHolder


TRAIN_HASH = '05ad85d871958a05c02ab51a4fde8530'
VALID_HASH = 'e53db4bff7dc4784123ae6df72e3b1f0'
TEST_HASH = '677b757ccec4809febd83850b43e1616'


def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


atom_encoder = {"C": 0, "N": 1, "S": 2, "O": 3, "F": 4, "Cl": 5, "Br": 6}
atom_decoder = ["C", "N", "S", "O", "F", "Cl", "Br"]


class MOSESDataset(InMemoryDataset):
    train_url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/train.csv'
    val_url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/test.csv'
    test_url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/test_scaffolds.csv'

    def __init__(self, split, root, transform=None, pre_transform=None, pre_filter=None):
        self.split = split
        if self.split == 'train':
            self.file_idx = 0
        elif self.split == 'val':
            self.file_idx = 1
        else:
            self.file_idx = 2
        self.atom_encoder = atom_encoder
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

        self.statistics = Statistics(num_nodes=load_pickle(self.processed_paths[1]),
                                     node_types=torch.from_numpy(np.load(self.processed_paths[2])).float(),
                                     bond_types=torch.from_numpy(np.load(self.processed_paths[3])).float(),
                                     valencies=load_pickle(self.processed_paths[4]))
        self.smiles = load_pickle(self.processed_paths[5])

    @property
    def raw_file_names(self):
        return ['train_moses.csv', 'val_moses.csv', 'test_moses.csv']

    @property
    def split_file_name(self):
        return ['train_moses.csv', 'val_moses.csv', 'test_moses.csv']

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        if self.split == 'train':
            return [f'train.pt', f'train_n.pickle', f'train_atom_types.npy', f'train_bond_types.npy',
                    f'train_valency.pickle', f'train_smiles.pickle']
        elif self.split == 'val':
            return [f'val.pt', f'val_n.pickle', f'val_atom_types.npy', f'val_bond_types.npy',
                    f'val_valency.pickle', f'val_smiles.pickle']
        else:
            return [f'test.pt', f'test_n.pickle', f'test_atom_types.npy', f'test_bond_types.npy',
                    f'test_valency.pickle', f'test_smiles.pickle']

    def download(self):
        import rdkit  # noqa
        train_path = download_url(self.train_url, self.raw_dir)
        os.rename(train_path, osp.join(self.raw_dir, 'train_moses.csv'))

        test_path = download_url(self.test_url, self.raw_dir)
        os.rename(test_path, osp.join(self.raw_dir, 'val_moses.csv'))

        valid_path = download_url(self.val_url, self.raw_dir)
        os.rename(valid_path, osp.join(self.raw_dir, 'test_moses.csv'))

    def process(self):
        RDLogger.DisableLog('rdApp.*')

        smile_list = pd.read_csv(self.split_paths[self.file_idx])['SMILES'].values

        data_list = []
        smiles_kept = []
        for i, smile in enumerate(tqdm(smile_list)):
            mol = Chem.MolFromSmiles(smile)
            if mol is not None:
                data = mol_to_torch_geometric(mol, atom_encoder, smile)
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                smiles_kept.append(smile)
        print(f"Number of smiles kept: {len(smiles_kept)} / {len(smile_list)}")

        statistics = compute_all_statistics(data_list, self.atom_encoder)

        save_pickle(statistics.num_nodes, self.processed_paths[1])
        np.save(self.processed_paths[2], statistics.node_types)
        np.save(self.processed_paths[3], statistics.bond_types)
        save_pickle(statistics.valencies, self.processed_paths[4])
        save_pickle(set(smiles_kept), self.processed_paths[5])
        torch.save(self.collate(data_list), self.processed_paths[0])


class MOSESDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(get_original_cwd()).parents[0]
        root_path = os.path.join(base_path, self.datadir)
        self.remove_h = False

        train_dataset = MOSESDataset(split='train', root=root_path)
        val_dataset = MOSESDataset(split='val', root=root_path)
        test_dataset = MOSESDataset(split='test', root=root_path)
        self.statistics = {'train': train_dataset.statistics, 'val': val_dataset.statistics,
                           'test': test_dataset.statistics}

        super().__init__(cfg, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset)


class MOSESinfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg):
        self.name = 'moses'
        self.is_molecular = True
        self.remove_h = False

        self.atom_encoder = atom_encoder
        self.atom_decoder = atom_decoder
        self.statistics = datamodule.statistics
        self.train_smiles = datamodule.train_dataset.smiles
        self.val_smiles = datamodule.val_dataset.smiles
        self.test_smiles = datamodule.test_dataset.smiles
        super().complete_infos(datamodule.statistics)

        self.input_dims = PlaceHolder(X=self.num_node_types, E=5, y=1)
        self.output_dims = PlaceHolder(X=self.num_node_types, E=5, y=0)
        self.max_weight = 9 * 80  # Quite arbitrary
        self.atom_weights = {0: 12, 1: 14, 2: 32, 3: 16, 4: 19, 5: 35.4, 6: 79.9}
        self.valencies = [4, 3, 2, 2, 1, 1, 1]
        self.rows = {0: 2, 1: 2, 2: 2, 3: 2, 4: 3, 5: 3, 6: 4}
        self.groups = {0: 14, 1: 15, 2: 16, 3: 17, 4: 16, 5: 17, 6: 17}
        self.families = {0: 0, 1: 0, 2: 0, 3: 8, 4: 0, 5: 8, 6: 8}
