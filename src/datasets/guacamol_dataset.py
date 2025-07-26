import os
import os.path as osp
import pathlib
import hashlib
from typing import Any, Sequence

import torch
from rdkit import Chem, RDLogger
from tqdm import tqdm
import numpy as np
from torch_geometric.data import InMemoryDataset, download_url
from hydra.utils import get_original_cwd

from datasets.dataset_utils import mol_to_torch_geometric, Statistics, load_pickle, save_pickle, to_list
from datasets.abstract_dataset import AbstractDatasetInfos, AbstractBucketDataModule
from metrics.metrics_utils import compute_all_statistics
from metrics.molecular_metrics import build_molecule, mol2smiles
from utils import PlaceHolder
from utils import to_dense

TRAIN_HASH = '05ad85d871958a05c02ab51a4fde8530'
VALID_HASH = 'e53db4bff7dc4784123ae6df72e3b1f0'
TEST_HASH = '677b757ccec4809febd83850b43e1616'


def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


def compare_hash(output_file: str, correct_hash: str) -> bool:
    """
    Computes the md5 hash of a SMILES file and check it against a given one
    Returns false if hashes are different
    """
    output_hash = hashlib.md5(open(output_file, 'rb').read()).hexdigest()
    if output_hash != correct_hash:
        print(f'{output_file} file has different hash, {output_hash}, than expected, {correct_hash}!')
        return False

    return True


atom_encoder = {
    "C": 0,
    "N": 1,
    "O": 2,
    "F": 3,
    "B": 4,
    "Br": 5,
    "Cl": 6,
    "I": 7,
    "P": 8,
    "S": 9,
    "Se": 10,
    "Si": 11,
}
atom_decoder = ["C", "N", "O", "F", "B", "Br", "Cl", "I", "P", "S", "Se", "Si"]


class GuacamolDataset(InMemoryDataset):
    train_url = ('https://figshare.com/ndownloader/files/13612760')
    test_url = 'https://figshare.com/ndownloader/files/13612757'
    valid_url = 'https://figshare.com/ndownloader/files/13612766'
    all_url = 'https://figshare.com/ndownloader/files/13612745'

    def __init__(self, split, root, filter_dataset, transform=None, pre_transform=None, pre_filter=None):
        self.split = split
        self.filter_dataset = filter_dataset
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
        return ['guacamol_v1_train.smiles', 'guacamol_v1_valid.smiles', 'guacamol_v1_test.smiles']

    @property
    def split_file_name(self):
        return ['guacamol_v1_train.smiles', 'guacamol_v1_valid.smiles', 'guacamol_v1_test.smiles']

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
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
        os.rename(train_path, osp.join(self.raw_dir, 'guacamol_v1_train.smiles'))
        train_path = osp.join(self.raw_dir, 'guacamol_v1_train.smiles')

        test_path = download_url(self.test_url, self.raw_dir)
        os.rename(test_path, osp.join(self.raw_dir, 'guacamol_v1_test.smiles'))
        test_path = osp.join(self.raw_dir, 'guacamol_v1_test.smiles')

        valid_path = download_url(self.valid_url, self.raw_dir)
        os.rename(valid_path, osp.join(self.raw_dir, 'guacamol_v1_valid.smiles'))
        valid_path = osp.join(self.raw_dir, 'guacamol_v1_valid.smiles')

        # check the hashes
        # Check whether the md5-hashes of the generated smiles files match
        # the precomputed hashes, this ensures everyone works with the same splits.
        valid_hashes = [
            compare_hash(train_path, TRAIN_HASH),
            compare_hash(valid_path, VALID_HASH),
            compare_hash(test_path, TEST_HASH),
        ]

        if not all(valid_hashes):
            raise SystemExit('Invalid hashes for the dataset files')

        print('Dataset download successful. Hashes are correct.')

        if files_exist(self.split_paths):
            return

    def process(self):
        RDLogger.DisableLog('rdApp.*')

        smile_list = open(self.split_paths[self.file_idx]).readlines()

        data_list = []
        smiles_kept = []
        for i, smile in enumerate(tqdm(smile_list)):
            mol = Chem.MolFromSmiles(smile)
            if mol is not None:
                data = mol_to_torch_geometric(mol, atom_encoder, smile)
                if self.filter_dataset:
                    dense_data = to_dense(data, x_classes=len(atom_decoder), e_classes=5)
                    dense_data = dense_data.collapse()
                    X, E = dense_data.X, dense_data.E

                    assert X.size(0) == 1
                    atom_types = X[0]
                    edge_types = E[0]

                    mol = build_molecule(atom_types, edge_types, atom_decoder)
                    smiles = mol2smiles(mol)
                    if smiles is not None:
                        try:
                            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                            if len(mol_frags) == 1:
                                data_list.append(data)
                                smiles_kept.append(smile)

                        except Chem.rdchem.AtomValenceException:
                            print("Valence error in GetmolFrags")
                        except Chem.rdchem.KekulizeException:
                            print("Can't kekulize molecule")
                else:
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


class GuacamolDataModule(AbstractBucketDataModule):
    def __init__(self, cfg):
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(get_original_cwd()).parents[0]
        root_path = os.path.join(base_path, self.datadir)
        self.filter = cfg.dataset.filter

        train_dataset = GuacamolDataset(split='train', root=root_path, filter_dataset=self.filter)
        val_dataset = GuacamolDataset(split='val', root=root_path, filter_dataset=self.filter)
        test_dataset = GuacamolDataset(split='test', root=root_path, filter_dataset=self.filter)

        self.statistics = {'train': train_dataset.statistics, 'val': val_dataset.statistics,
                           'test': test_dataset.statistics}
        self.remove_h = cfg.dataset.remove_h
        super().__init__(cfg, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset)


class Guacamolinfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg):
        self.name = 'guacamol'
        self.is_molecular = True
        self.remove_h = cfg.dataset.remove_h

        self.atom_encoder = atom_encoder
        self.atom_decoder = atom_decoder
        self.statistics = datamodule.statistics
        self.train_smiles = datamodule.train_dataset.smiles
        self.val_smiles = datamodule.val_dataset.smiles
        self.test_smiles = datamodule.test_dataset.smiles
        super().complete_infos(datamodule.statistics)

        self.input_dims = PlaceHolder(X=self.num_node_types, E=5, y=1)
        self.output_dims = PlaceHolder(X=self.num_node_types, E=5, y=0)
        self.max_weight = 1000
        self.atom_weights = {1: 12, 2: 14, 3: 16, 4: 19, 5: 10.81, 6: 79.9,
                             7: 35.45, 8: 126.9, 9: 30.97, 10: 30.07, 11: 78.97, 12: 28.09}
        self.valencies = [4, 3, 2, 1, 3, 1, 1, 1, 3, 2, 2, 4]



