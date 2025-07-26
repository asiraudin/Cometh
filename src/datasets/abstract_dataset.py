from torch.utils.data import Subset
from typing import Optional

import torch
from datasets.bucket_loader import BucketLightningDataset
try:
    from torch_geometric.data import LightningDataset
except ImportError:
    from torch_geometric.data.lightning import LightningDataset
from diffusion.distributions import DistributionNodes
from torch_geometric.loader import DataLoader


def maybe_subset(ds, random_subset: Optional[float]=None,split=None) -> torch.utils.data.Dataset:
    if random_subset is None or split in {"test", "val"}:
        return ds
    else:
        idx = torch.randperm(len(ds))[:int(random_subset * len(ds))]
        return Subset(ds, idx)


class AbstractDataModule(LightningDataset):
    def __init__(self, cfg, train_dataset, val_dataset, test_dataset):
        super().__init__(train_dataset, val_dataset, test_dataset, batch_size=cfg.train.batch_size,
                         num_workers=cfg.train.num_workers, shuffle='debug' not in cfg.general.name,
                         pin_memory=getattr(cfg.dataset, "pin_memory", False))
        self.cfg = cfg

    def prepare_dataloader(self):
        self.dataloaders = {}
        self.dataloaders["train"] = self.train_dataloader()
        self.dataloaders["val"] = self.val_dataloader()
        self.dataloaders["test"] = self.test_dataloader()


class AbstractBucketDataModule(BucketLightningDataset):
    def __init__(self, cfg, train_dataset, val_dataset, test_dataset):
        super().__init__(train_dataset, val_dataset, test_dataset, batch_size=cfg.train.batch_size,
                         num_workers=cfg.train.num_workers, thresholds=cfg.dataset.thresholds,
                         shuffle='debug' not in cfg.general.name,
                         pin_memory=getattr(cfg.dataset, "pin_memory", False))
        self.cfg = cfg

    def prepare_dataloader(self):
        self.dataloaders = {}
        self.dataloaders["train"] = self.train_dataloader()
        self.dataloaders["val"] = self.val_dataloader()
        self.dataloaders["test"] = self.test_dataloader()


class AbstractDatasetInfos:
    def complete_infos(self, statistics):
        # atom and edge type information
        self.node_types = statistics["train"].node_types
        self.edge_types = statistics["train"].bond_types
        self.num_node_types = len(self.node_types)
        self.num_edge_types = len(self.edge_types)

        # Train + val + test for n_nodes
        train_n_nodes = statistics["train"].num_nodes
        val_n_nodes = statistics["val"].num_nodes
        test_n_nodes = statistics["test"].num_nodes
        max_n_nodes = max(
            max(train_n_nodes.keys()), max(val_n_nodes.keys()), max(test_n_nodes.keys())
        )
        n_nodes = torch.zeros(max_n_nodes + 1, dtype=torch.long)
        for c in [train_n_nodes, val_n_nodes, test_n_nodes]:
            for key, value in c.items():
                n_nodes[key] += value
        self.n_nodes = n_nodes / n_nodes.sum()

        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)
