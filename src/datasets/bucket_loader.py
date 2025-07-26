from collections.abc import Mapping, Sequence
from typing import Union, List, Optional
import math

import torch
from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch.utils.data.dataloader import default_collate
import torch.utils.data
from torch_geometric.loader import DataLoader
try:
    from torch_geometric.data import LightningDataset
except ImportError:
    from torch_geometric.data.lightning import LightningDataset

THRESH_DETAILS = [30, 50, 90]


class BucketCollater:
    def __init__(self, follow_batch, exclude_keys, bucket_thresholds=None):
        """ Copypaste from pyg.loader.Collater + small changes
            Credits to Igor Krawczuk

        """
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        if bucket_thresholds is None:
            bucket_thresholds = THRESH_DETAILS
        self.bucket_thresholds = sorted(bucket_thresholds)

    def __call__(self, batch):
        # checks the number of node for basedata graphs and slots into appropriate buckets,
        # errors on other options
        elem = batch[0]
        if isinstance(elem, BaseData):
            buckets = {k: [] for k in self.bucket_thresholds}
            buckets[None] = []

            for e in batch:
                e: BaseData
                for k in self.bucket_thresholds:
                    nn = e.num_nodes
                    if k is not None and nn <= k:
                        buckets[k].append(e)
                        break
            else:
                # obscure python feature: if we early exit, it's the "if" branch, if we run to completion
                # i.e. the graph has more nodes than the largest threshold, we slot things into this final bucket
                buckets[None].append(e)
            batches = [Batch.from_data_list(buckets[k],
                                            self.follow_batch,
                                            self.exclude_keys) if len(buckets[k]) > 0
                       else None for k in self.bucket_thresholds]

            return batches

        elif True:
            # early exit
            raise NotImplementedError("Only supporting BaseData for now")
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f'DataLoader found invalid type: {type(elem)}')

    def collate(self, batch):  # Deprecated...
        return self(batch)


class BucketDataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` into mini-batches, each minibatch being a bucket with num_nodes < some threshold,
    except the last which holds the overflow-graphs. Apart from the bucketing, identical to torch_geometric.loader.DataLoader
    Default bucket_thresholds is [30,50,90], yielding 4 buckets
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
            self,
            dataset: Union[Dataset, List[BaseData]],
            batch_size: int = 1,
            shuffle: bool = False,
            follow_batch: Optional[List[str]] = None,
            exclude_keys: Optional[List[str]] = None,
            bucket_thresholds: Optional[List[int]] = None,
            **kwargs,
    ):

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        # Save for PyTorch Lightning:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=BucketCollater(follow_batch, exclude_keys, bucket_thresholds=bucket_thresholds),
            **kwargs,
        )


class BucketLightningDataset(LightningDataset):
    r"""Converts a set of :class:`~torch_geometric.data.Dataset` objects into a
    :class:`pytorch_lightning.LightningDataModule` variant, which can be
    automatically used as a :obj:`datamodule` for multi-GPU graph-level
    training via `PyTorch Lightning <https://www.pytorchlightning.ai>`__.
    :class:`LightningDataset` will take care of providing mini-batches via
    :class:`~torch_geometric.loader.DataLoader`.

    .. note::

        Currently only the
        :class:`pytorch_lightning.strategies.SingleDeviceStrategy` and
        :class:`pytorch_lightning.strategies.DDPSpawnStrategy` training
        strategies of `PyTorch Lightning
        <https://pytorch-lightning.readthedocs.io/en/latest/guides/
        speed.html>`__ are supported in order to correctly share data across
        all devices/processes:

        .. code-block::

            import pytorch_lightning as pl
            trainer = pl.Trainer(strategy="ddp_spawn", accelerator="gpu",
                                 devices=4)
            trainer.fit(model, datamodule)

    Args:
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset, optional): The validation dataset.
            (default: :obj:`None`)
        test_dataset (Dataset, optional): The test dataset.
            (default: :obj:`None`)
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        num_workers: How many subprocesses to use for data loading.
            :obj:`0` means that the data will be loaded in the main process.
            (default: :obj:`0`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.loader.DataLoader`.
    """
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        thresholds: Optional[List] = None,
        **kwargs,
    ):
        super().__init__(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            # has_val=val_dataset is not None,
            # has_test=test_dataset is not None,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.thresholds = thresholds

    def dataloader(self, dataset: Dataset, shuffle: bool = False, **kwargs) -> BucketDataLoader:
        return BucketDataLoader(dataset, shuffle=shuffle, bucket_thresholds=self.thresholds, **self.kwargs)
