from rdkit import Chem, RDLogger
import os
import re
from collections import Counter

import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import wandb
from torchmetrics import MeanMetric, MaxMetric

from utils import NoSyncMetric as Metric, NoSyncMetricCollection as MetricCollection
from metrics.metrics_utils import counter_to_tensor, wasserstein1d, total_variation1d


class SamplingMolecularMetrics(nn.Module):
    def __init__(self, train_smiles, dataset_infos, test):
        super().__init__()
        self.dataset_infos = dataset_infos
        self.atom_decoder = dataset_infos.atom_decoder

        self.train_smiles = train_smiles
        self.test = test

        self.atom_stable = MeanMetric()
        self.mol_stable = MeanMetric()

        # Retrieve dataset smiles only for qm9 currently.
        self.train_smiles = set(train_smiles)
        self.validity_metric = MeanMetric()
        self.relaxed_validity_metric = MeanMetric()
        self.uniqueness = MeanMetric()
        self.novelty = MeanMetric()
        self.valency_w1 = MeanMetric()

    def reset(self):
        for metric in [self.atom_stable, self.mol_stable, self.validity_metric, self.uniqueness,
                       self.novelty]:
            metric.reset()

    def compute_validity(self, generated):
        valid = []
        all_smiles = []
        error_message = Counter()

        for i, mol in enumerate(generated):
            """generated: list of couples (positions, node_types)"""
            rdmol = mol.rdkit_mol
            if rdmol is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(
                        rdmol, asMols=True, sanitizeFrags=True
                    )
                    if len(mol_frags) > 1:
                        error_message[4] += 1
                    largest_mol = max(
                        mol_frags, default=mol, key=lambda m: m.GetNumAtoms()
                    )
                    Chem.SanitizeMol(largest_mol)
                    smiles = Chem.MolToSmiles(largest_mol)
                    valid.append(smiles)
                    all_smiles.append(smiles)
                    error_message[-1] += 1
                except Chem.rdchem.AtomValenceException:
                    error_message[1] += 1
                    all_smiles.append('error')
                except Chem.rdchem.KekulizeException:
                    error_message[2] += 1
                    all_smiles.append('error')
                except Chem.rdchem.AtomKekulizeException or ValueError:
                    error_message[3] += 1
                    all_smiles.append('error')
        print(
            f"Error messages: AtomValence {error_message[1]}, Kekulize {error_message[2]}, other {error_message[3]}, "
            f" -- No error {error_message[-1]}"
        )
        self.validity_metric.update(
            value=len(valid) / len(generated), weight=len(generated)
        )
        return valid, all_smiles, error_message

    def evaluate(self, generated, local_rank):
        # Validity
        valid, all_smiles, _ = self.compute_validity(generated)
        validity = self.validity_metric.compute().item()
        uniqueness, novelty = 0, 0

        # Uniqueness
        if len(valid) > 0:
            unique = list(set(valid))
            self.uniqueness.update(value=len(unique) / len(valid), weight=len(valid))
            uniqueness = self.uniqueness.compute()
            if self.train_smiles is not None:
                novel = []
                for smiles in unique:
                    if smiles not in self.train_smiles:
                        novel.append(smiles)
                self.novelty.update(value=len(novel) / len(unique), weight=len(unique))
            novelty = self.novelty.compute()

        if local_rank == 0:
            num_molecules = int(self.validity_metric.weight.detach())
            print(f"Validity over {num_molecules} molecules:"
                  f" {validity * 100 :.2f}%")
            print(f"Uniqueness: {uniqueness * 100 :.2f}% WARNING: do not trust this metric on multi-gpu")
            print(f"Novelty: {novelty * 100 :.2f}%")

        key = "val" if not self.test else "test"
        dic = {
            f"{key}/Validity": validity * 100,
            f"{key}/Uniqueness": uniqueness * 100 if uniqueness != 0 else 0,
            f"{key}/Novelty": novelty * 100 if novelty != 0 else 0,
        }

        if wandb.run:
            wandb.log(dic, commit=False)
        return all_smiles if len(all_smiles) > 0 else [], dic

    def __call__(self, generated_graphs: list, current_epoch, local_rank):
        molecules = []
        for graph in generated_graphs:
            molecule = Molecule(atom_types=graph[0], bond_types=graph[1], atom_decoder=self.atom_decoder)
            molecules.append(molecule)

        # Atom and molecule stability
        if not self.dataset_infos.remove_h:
            print(f'Analyzing molecule stability on {local_rank}...')
            for i, mol in enumerate(molecules):
                mol_stable, at_stable, num_bonds = check_stability(mol, self.dataset_infos)
                self.mol_stable.update(value=mol_stable)
                self.atom_stable.update(value=at_stable / num_bonds, weight=num_bonds)
            key = "val" if not self.test else "test"
            stability_dict = {f"{key}/mol_stable": self.mol_stable.compute().item(),
                              f"{key}/  atm_stable": self.atom_stable.compute().item()}
            if local_rank == 0:
                print("Stability metrics:", stability_dict)
                if wandb.run:
                    wandb.log(stability_dict, commit=False)

        # Validity, uniqueness, novelty. Returned smiles are the valid and unique ones.
        all_generated_smiles, metrics = self.evaluate(molecules, local_rank=local_rank)
        # Save in any case in the graphs folder
        os.makedirs('graphs', exist_ok=True)
        textfile = open(f'graphs/valid_unique_molecules_e{current_epoch}_GR{local_rank}.txt', "w")
        textfile.writelines(all_generated_smiles)
        textfile.close()
        # Save in the root folder if test_model
        if self.test:
            filename = f'final_smiles_GR{local_rank}_{0}.txt'
            for i in range(2, 10):
                if os.path.exists(filename):
                    filename = f'final_smiles_GR{local_rank}_{i}.txt'
                else:
                    break
            with open(filename, 'w') as fp:
                for smiles in all_generated_smiles:
                    # write each item on a new line
                    fp.write("%s\n" % smiles)
                print(f'All smiles saved on rank {local_rank}')

        # Compute statistics
        stat = self.dataset_infos.statistics['test'] if self.test else self.dataset_infos.statistics['val']

        valency_w1, valency_w1_per_class = valency_distance(molecules, stat.valencies, stat.node_types,
                                                            self.dataset_infos.atom_encoder)
        self.valency_w1(valency_w1)

        key = "val" if not self.test else "test"
        metrics[f'{key}/ValencyW1'] = self.valency_w1.compute().detach()
        # if local_rank == 0:
        #     print(f"Sampling metrics", {k: round(val, 3) for k, val in metrics.items()})

        if wandb.run:
            wandb.log(metrics, commit=False)
        print(f"Sampling molecular metrics done on {local_rank}.")
        return metrics


def valency_distance(molecules, target_valencies, atom_types_probabilities, atom_encoder):
    # Build a dict for the generated molecules that is similar to the target one
    num_atom_types = len(atom_types_probabilities)
    generated_valencies = {i: Counter() for i in range(num_atom_types)}
    for molecule in molecules:
        edge_types = molecule.bond_types
        edge_types[edge_types == 4] = 1.5
        valencies = torch.sum(edge_types, dim=0)
        for atom, val in zip(molecule.atom_types, valencies):
            generated_valencies[atom.item()][val.item()] += 1

    # Convert the valencies to a tensor of shape (num_atom_types, max_valency)
    max_valency_target = max(max(vals.keys()) if len(vals) > 0 else -1 for vals in target_valencies.values())
    max_valency_generated = max(max(vals.keys()) if len(vals) > 0 else -1 for vals in generated_valencies.values())
    max_valency = max(max_valency_target, max_valency_generated)

    valencies_target_tensor = torch.zeros(num_atom_types, max_valency + 1)
    for atom_type, valencies in target_valencies.items():
        for valency, count in valencies.items():
            valencies_target_tensor[atom_encoder[atom_type], valency] = count

    valencies_generated_tensor = torch.zeros(num_atom_types, max_valency + 1)
    for atom_type, valencies in generated_valencies.items():
        for valency, count in valencies.items():
            valencies_generated_tensor[atom_type, valency] = count

    # Normalize the distributions
    s1 = torch.sum(valencies_target_tensor, dim=1, keepdim=True)
    s1[s1 == 0] = 1
    valencies_target_tensor = valencies_target_tensor / s1

    s2 = torch.sum(valencies_generated_tensor, dim=1, keepdim=True)
    s2[s2 == 0] = 1
    valencies_generated_tensor = valencies_generated_tensor / s2

    cs_target = torch.cumsum(valencies_target_tensor, dim=1)
    cs_generated = torch.cumsum(valencies_generated_tensor, dim=1)

    w1_per_class = torch.sum(torch.abs(cs_target - cs_generated), dim=1)

    total_w1 = torch.sum(w1_per_class * atom_types_probabilities).item()
    return total_w1, w1_per_class


class CEPerClass(Metric):
    full_state_update = True

    def __init__(self, class_id):
        super().__init__()
        self.class_id = class_id
        self.add_state('total_ce', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.softmax = torch.nn.Softmax(dim=-1)
        self.binary_cross_entropy = torch.nn.BCELoss(reduction='sum')

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.
        Args:
            preds: Predictions from model   (bs, n, d) or (bs, n, n, d)
            target: Ground truth values     (bs, n, d) or (bs, n, n, d)
        """
        target = target.reshape(-1, target.shape[-1])
        mask = (target != 0.).any(dim=-1)

        prob = self.softmax(preds)[..., self.class_id]
        prob = prob.flatten()[mask]

        target = target[:, self.class_id]
        target = target[mask]

        output = self.binary_cross_entropy(prob, target)
        self.total_ce += output
        self.total_samples += prob.numel()

    def compute(self):
        return self.total_ce / self.total_samples


class MeanNumberEdge(Metric):
    full_state_update = True

    def __init__(self):
        super().__init__()
        self.add_state('total_edge', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, molecules, weight=1.0) -> None:
        for molecule in molecules:
            _, edge_types = molecule
            triu_edge_types = torch.triu(edge_types, diagonal=1)
            bonds = torch.nonzero(triu_edge_types)
            self.total_edge += len(bonds)
        self.total_samples += len(molecules)

    def compute(self):
        return self.total_edge / self.total_samples


class HydrogenCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class CarbonCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class NitroCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class OxyCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class FluorCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class BoronCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class BrCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class ClCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class IodineCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class PhosphorusCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SulfurCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SeCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SiCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class NoBondCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SingleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class DoubleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class TripleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AromaticCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AtomMetricsCE(MetricCollection):
    def __init__(self, dataset_infos):
        atom_decoder = dataset_infos.atom_decoder

        class_dict = {'H': HydrogenCE, 'C': CarbonCE, 'N': NitroCE, 'O': OxyCE, 'F': FluorCE, 'B': BoronCE,
                      'Br': BrCE, 'Cl': ClCE, 'I': IodineCE, 'P': PhosphorusCE, 'S': SulfurCE, 'Se': SeCE,
                      'Si': SiCE}

        metrics_list = []
        for i, atom_type in enumerate(atom_decoder):
            metrics_list.append(class_dict[atom_type](i))
        super().__init__(metrics_list)


class BondMetricsCE(MetricCollection):
    def __init__(self):
        ce_no_bond = NoBondCE(0)
        ce_SI = SingleCE(1)
        ce_DO = DoubleCE(2)
        ce_TR = TripleCE(3)
        ce_AR = AromaticCE(4)
        super().__init__([ce_no_bond, ce_SI, ce_DO, ce_TR, ce_AR])


class TrainMolecularMetrics(nn.Module):
    def __init__(self, dataset_infos):
        super().__init__()
        self.train_atom_metrics = AtomMetricsCE(dataset_infos=dataset_infos)
        self.train_bond_metrics = BondMetricsCE()

    def forward(self, masked_pred, masked_true, log: bool):
        self.train_atom_metrics(masked_pred.X, masked_true.X)
        self.train_bond_metrics(masked_pred.E, masked_true.E)
        if not log:
            return

        to_log = {}
        for key, val in self.train_atom_metrics.compute().items():
            to_log['train/' + key] = val.detach()
        for key, val in self.train_bond_metrics.compute().items():
            to_log['train/' + key] = val.detach()
        if wandb.run:
            wandb.log(to_log, commit=False)
        return to_log

    def reset(self):
        for metric in [self.train_atom_metrics, self.train_bond_metrics]:
            metric.reset()

    def log_epoch_metrics(self, current_epoch, local_rank):
        epoch_atom_metrics = self.train_atom_metrics.compute()
        epoch_bond_metrics = self.train_bond_metrics.compute()

        to_log = {}
        for key, val in epoch_atom_metrics.items():
            to_log['train_epoch/' + key] = val.detach()
        for key, val in epoch_bond_metrics.items():
            to_log['train_epoch/' + key] = val.detach()

        if wandb.run:
            wandb.log(to_log, commit=False)

        for key, val in epoch_atom_metrics.items():
            epoch_atom_metrics[key] = round(val.detach(), 3)
        for key, val in epoch_bond_metrics.items():
            epoch_bond_metrics[key] = round(val.detach(), 3)
        print(f"Epoch {current_epoch} on rank {local_rank}: {epoch_atom_metrics} -- {epoch_bond_metrics}")

        return to_log


allowed_bonds = {'H': {0: 1, 1: 0, -1: 0},
                 'C': {0: [3, 4], 1: 3, -1: 3},
                 'N': {0: [2, 3], 1: [2, 3, 4], -1: 2},    # In QM9, N+ seems to be present in the form NH+ and NH2+
                 'O': {0: 2, 1: 3, -1: 1},
                 'F': {0: 1, -1: 0},
                 'B': 3, 'Al': 3, 'Si': 4,
                 'P': {0: [3, 5], 1: 4},
                 'S': {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
                 'Cl': 1, 'As': 3,
                 'Br': {0: 1, 1: 2}, 'I': 1, 'Hg': [1, 2], 'Bi': [3, 5], 'Se': [2, 4, 6]}
bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
             Chem.rdchem.BondType.AROMATIC]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}


class Molecule:
    def __init__(self, atom_types, bond_types, atom_decoder):
        """ atom_types: n      LongTensor
            charges: n         LongTensor
            bond_types: n x n  LongTensor
            positions: n x 3   FloatTensor
            atom_decoder: extracted from dataset_infos. """
        assert atom_types.dim() == 1 and atom_types.dtype == torch.long, f"shape of atoms {atom_types.shape} " \
                                                                         f"and dtype {atom_types.dtype}"
        assert bond_types.dim() == 2 and bond_types.dtype == torch.long, f"shape of bonds {bond_types.shape} --" \
                                                                         f" {bond_types.dtype}"
        assert len(atom_types.shape) == 1
        assert len(bond_types.shape) == 2

        self.atom_types = atom_types.long()
        self.bond_types = bond_types.long()

        self.rdkit_mol = build_molecule(self.atom_types, self.bond_types, atom_decoder)
        self.num_nodes = len(atom_types)
        self.num_atom_types = len(atom_decoder)


def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


def check_stability(mol, dataset_info, debug=False, atom_decoder=None):
    atom_types = mol.atom_types
    bond_types = mol.bond_types
    if atom_decoder is None:
        atom_decoder = dataset_info.atom_decoder

    n_bonds = np.zeros(len(atom_types), dtype='int')

    for i in range(len(atom_types)):
        for j in range(i + 1, len(atom_types)):
            n_bonds[i] += abs((bond_types[i, j] + bond_types[j, i])/2)
            n_bonds[j] += abs((bond_types[i, j] + bond_types[j, i])/2)
    n_stable_bonds = 0
    for atom_type, atom_n_bond in zip(atom_types, n_bonds):
        possible_bonds = allowed_bonds[atom_decoder[atom_type]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == atom_n_bond
        else:
            is_stable = atom_n_bond in possible_bonds
        if not is_stable and debug:
            print("Invalid bonds for molecule %s with %d bonds" % (atom_decoder[atom_type], atom_n_bond))
        n_stable_bonds += int(is_stable)

    molecule_stable = n_stable_bonds == len(atom_types)
    return molecule_stable, n_stable_bonds, len(atom_types)


def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence


def build_molecule(atom_types, edge_types, atom_decoder, verbose=False):
    if verbose:
        print("\nbuilding new molecule")

    mol = Chem.RWMol()
    for atom in atom_types:
        if atom == -1:
            continue
        a = Chem.Atom(atom_decoder[int(atom.detach())])
        mol.AddAtom(a)
        if verbose:
            print("Atom added: ", atom.detach(), atom_decoder[atom.detach()])

    edge_types = torch.triu(edge_types)
    edge_types[edge_types == -1] = 0
    all_bonds = torch.nonzero(edge_types)
    for i, bond in enumerate(all_bonds):
        if int(bond[0].detach()) != int(bond[1].detach()):
            mol.AddBond(
                int(bond[0].detach()),
                int(bond[1].detach()),
                bond_dict[edge_types[int(bond[0]), int(bond[1])].detach()]
            )
            if verbose:
                print("bond added:",
                      int(bond[0].detach()),
                      int(bond[1].detach()),
                      edge_types[int(bond[0]), int(bond[1])].detach(),
                      bond_dict[edge_types[int(bond[0]), int(bond[1])].detach()])
            # add formal charge to atom: e.g. [O+], [N+], [S+]
            # not support [O-], [N-], [S-], [NH+] etc.
            flag, atomid_valence = check_valency(mol)
            if verbose:
                print("flag, valence", flag, atomid_valence)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if verbose:
                    print("atomic num of atom with a large valence", an)
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
                    # print("Formal charge added")
    return mol


