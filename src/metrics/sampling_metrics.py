from collections import Counter

import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import wandb
from torchmetrics import MeanMetric, MaxMetric, Metric, MeanAbsoluteError
import torch
from torch import Tensor
from metrics.metrics_utils import (
    counter_to_tensor,
    wasserstein1d,
    total_variation1d,
)


class SamplingMetrics(nn.Module):
    def __init__(self, dataset_infos, test, dataloaders=None):
        super().__init__()
        self.dataset_infos = dataset_infos
        self.test = test

        self.disconnected = MeanMetric()
        self.mean_components = MeanMetric()
        self.max_components = MaxMetric()
        self.num_nodes_w1 = MeanMetric()
        self.node_types_tv = MeanMetric()
        self.edge_types_tv = MeanMetric()

        self.domain_metrics = None
        if dataset_infos.is_molecular:
            from metrics.molecular_metrics import (
                SamplingMolecularMetrics,
            )

            self.domain_metrics = SamplingMolecularMetrics(
                dataset_infos.train_smiles,
                dataset_infos,
                test
            )

        elif dataset_infos.spectre:
            from metrics.spectre_utils import (
                Comm20SamplingMetrics,
                PlanarSamplingMetrics,
                SBMSamplingMetrics,
                ProteinSamplingMetrics,
                EgoSamplingMetrics
            )

            if dataset_infos.name == "comm20":
                self.domain_metrics = Comm20SamplingMetrics(dataloaders=dataloaders)
            elif dataset_infos.name == "planar":
                self.domain_metrics = PlanarSamplingMetrics(dataloaders=dataloaders)
            elif dataset_infos.name == "sbm":
                self.domain_metrics = SBMSamplingMetrics(dataloaders=dataloaders)
            elif dataset_infos.name == "protein":
                self.domain_metrics = ProteinSamplingMetrics(dataloaders=dataloaders)
            elif dataset_infos.name == "ego":
                self.domain_metrics = EgoSamplingMetrics(dataloaders=dataloaders)
            else:
                raise ValueError(
                    "Dataset {} not implemented".format(dataset_infos.dataset_name)
                )

    def reset(self):
        for metric in [
            self.mean_components,
            self.max_components,
            self.disconnected,
            self.num_nodes_w1,
            self.node_types_tv,
            self.edge_types_tv,
        ]:
            metric.reset()
        if self.domain_metrics is not None:
            self.domain_metrics.reset()

    def compute_all_metrics(self, generated_graphs: list, current_epoch, local_rank):
        """Compare statistics of the generated data with statistics of the val/test set"""
        stat = (
            self.dataset_infos.statistics["test"]
            if self.test
            else self.dataset_infos.statistics["val"]
        )
        to_log = {}
        if self.domain_metrics is not None:
            do_metrics = self.domain_metrics(
                generated_graphs, current_epoch, local_rank
            )
            to_log.update(do_metrics)


        # Number of nodes
        self.num_nodes_w1(number_nodes_distance(generated_graphs, stat.num_nodes))

        # Node types
        node_type_tv, node_tv_per_class = node_types_distance(
            generated_graphs, stat.node_types, save_histogram=True
        )
        self.node_types_tv(node_type_tv)

        # Edge types
        edge_types_tv, edge_tv_per_class = edge_types_distance(
            generated_graphs, stat.bond_types, save_histogram=True
        )
        self.edge_types_tv(edge_types_tv)

        # Components
        device = self.disconnected.device
        connected_comp = connected_components(generated_graphs).to(device)
        self.disconnected(connected_comp > 1)
        self.mean_components(connected_comp)
        self.max_components(connected_comp)

        key = "val" if not self.test else "test"
        to_log.update({
            f"{key}/NumNodesW1": self.num_nodes_w1.compute().detach(),
            f"{key}/NodeTypesTV": self.node_types_tv.compute().detach(),
            f"{key}/EdgeTypesTV": self.edge_types_tv.compute().detach(),
            f"{key}/Disconnected": self.disconnected.compute().detach() * 100,
            f"{key}/MeanComponents": self.mean_components.compute().detach(),
            f"{key}/MaxComponents": self.max_components.compute().detach(),
        })

        if wandb.run:
            wandb.log(to_log, commit=False)
        if local_rank == 0:
            print(
                f"Sampling metrics", {
                    key: round(val.item(), 5) if isinstance(val, Tensor)
                    else round(val, 5)
                    for key, val in to_log.items()}
            )

        return to_log


def number_nodes_distance(generated_graphs, dataset_counts):
    max_number_nodes = max(dataset_counts.keys())
    reference_n = torch.zeros(
        max_number_nodes + 1
    )
    for n, count in dataset_counts.items():
        reference_n[n] = count

    c = Counter()
    for graph in generated_graphs:
        c[len(graph[0])] += 1

    generated_n = counter_to_tensor(c)
    return wasserstein1d(generated_n, reference_n)


def node_types_distance(generated_graphs, target, save_histogram=True):
    generated_distribution = torch.zeros_like(target)

    for graph in generated_graphs:
        for atom_type in graph[0]:
            generated_distribution[atom_type] += 1

    if save_histogram:
        if wandb.run:
            data = [[k, l] for k, l in zip(target, generated_distribution/generated_distribution.sum())]
            table = wandb.Table(data=data, columns=["target", "generate"])
            wandb.log({'node distribution': wandb.plot.histogram(table, 'types', title="node distribution")})

        np.save("generated_node_types.npy", generated_distribution.cpu().numpy())

    return total_variation1d(generated_distribution, target)


def edge_types_distance(generated_graphs, target, save_histogram=True):
    device = generated_graphs[0][1].device
    generated_distribution = torch.zeros_like(target).to(device)

    for graph in generated_graphs:
        edge_types = graph[1].clone()
        mask = torch.ones_like(edge_types)
        mask = torch.triu(mask, diagonal=1).bool()
        edge_types = edge_types[mask]
        unique_edge_types, counts = torch.unique(edge_types, return_counts=True)
        for type, count in zip(unique_edge_types, counts):
            generated_distribution[type] += count

    if save_histogram:
        if wandb.run:
            data = [[k, l] for k, l in zip(target, generated_distribution/generated_distribution.sum())]
            table = wandb.Table(data=data, columns=["target", "generate"])
            wandb.log({'edge distribution': wandb.plot.histogram(table, 'types', title="edge distribution")})

        np.save("generated_bond_types.npy", generated_distribution.cpu().numpy())

    tv, tv_per_class = total_variation1d(generated_distribution, target.to(device))
    return tv, tv_per_class


def connected_components(generated_graphs):
    num_graphs = len(generated_graphs)
    all_num_components = torch.zeros(num_graphs)
    for i in range(num_graphs):
        adj_matrix = sp.csr_matrix(np.array(generated_graphs[i][1].cpu()))
        num_components, component = sp.csgraph.connected_components(adj_matrix)
        all_num_components[i] = num_components

    return all_num_components


class HistogramsMAE(MeanAbsoluteError):
    def __init__(self, target_histogram, **kwargs):
        """Compute the distance between histograms."""
        super().__init__(**kwargs)
        assert (target_histogram.sum() - 1).abs() < 1e-3
        self.target_histogram = target_histogram

    def update(self, pred):
        pred = pred / pred.sum()
        target = self.target_histogram.to(pred.device)
        super().update(pred, target)


class MeanNumberEdge(Metric):
    full_state_update = True

    def __init__(self):
        super().__init__()
        self.add_state("total_edge", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, molecules, weight=1.0) -> None:
        for molecule in molecules:
            _, edge_types = molecule
            triu_edge_types = torch.triu(edge_types, diagonal=1)
            bonds = torch.nonzero(triu_edge_types)
            self.total_edge += len(bonds)
        self.total_samples += len(molecules)

    def compute(self):
        return self.total_edge / self.total_samples
