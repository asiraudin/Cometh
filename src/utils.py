import os

import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch, remove_self_loops
from torchmetrics import Metric, MetricCollection, KLDivergence
from omegaconf import OmegaConf
import wandb


class NoSyncMetricCollection(MetricCollection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs) #disabling syncs since it messes up DDP sub-batching


class NoSyncMetric(Metric):
    def __init__(self):
        super().__init__(sync_on_compute=False,dist_sync_on_step=False) #disabling syncs since it messes up DDP sub-batching


class NoSyncKL(KLDivergence):
    def __init__(self):
        super().__init__(sync_on_compute=False,dist_sync_on_step=False) #disabling syncs since it messes up DDP sub-batching


def create_folders(args):
    try:
        # os.makedirs('checkpoints')
        os.makedirs('graphs')
        os.makedirs('chains')
    except OSError:
        pass

    try:
        # os.makedirs('checkpoints/' + args.general.name)
        os.makedirs('graphs/' + args.general.name)
        os.makedirs('chains/' + args.general.name)
    except OSError:
        pass


def to_one_hot(X, E, x_classes, e_classes, node_mask):
    X = F.one_hot(X, num_classes=x_classes).float() if x_classes > 1 else X.float()
    E = F.one_hot(E, num_classes=e_classes).float()
    placeholder = PlaceHolder(X=X, E=E, y=None)
    pl = placeholder.mask(node_mask)
    return pl.X, pl.E


def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E


def to_dense(data, x_classes, e_classes, device=None):
    X, node_mask = to_dense_batch(x=data.x, batch=data.batch)
    max_num_nodes = X.size(1)
    edge_index, edge_attr = remove_self_loops(data.edge_index, data.edge_attr)
    E = to_dense_adj(edge_index=edge_index, batch=data.batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)

    X, E = to_one_hot(X=X, E=E, x_classes=x_classes, e_classes=e_classes, node_mask=node_mask)
    y = X.new_zeros((X.shape[0], 0))

    if device is not None:
        X = X.to(device)
        E = E.to(device)
        y = y.to(device)
        node_mask = node_mask.to(device)
    data = PlaceHolder(X=X, E=E, y=y,  node_mask=node_mask).mask()
    if x_classes == 1:
        data = data.permutate_adj()
    return data


def cumsum(x, dim: int = 0):
    r"""Returns the cumulative sum of elements of :obj:`x`.
    In contrast to :meth:`torch.cumsum`, prepends the output with zero.


    Taken from PyG 2.3
    Args:
        x (torch.Tensor): The input tensor.
        dim (int, optional): The dimension to do the operation over.
            (default: :obj:`0`)

    """
    size = x.size()[:dim] + (x.size(dim) + 1, ) + x.size()[dim + 1:]
    out = x.new_empty(size)

    out.narrow(dim, 0, 1).zero_()
    torch.cumsum(x, dim=dim, out=out.narrow(dim, 1, x.size(dim)))

    return out


def dense_to_sparse_(adj, mask):
    if adj.dim() < 2 or adj.dim() > 3:
        raise ValueError(f"Dense adjacency matrix 'adj' must be two- or "
                         f"three-dimensional (got {adj.dim()} dimensions)")

    if mask is not None and adj.dim() == 2:
        print("Mask should not be provided in case the dense adjacency matrix is two-dimensional")
        mask = None

    if mask is not None and mask.dim() != 2:
        raise ValueError(f"Mask must be two-dimensional "
                         f"(got {mask.dim()} dimensions)")

    if mask is not None and adj.size(-2) != adj.size(-1):
        raise ValueError(f"Mask is only supported on quadratic adjacency "
                         f"matrices (got [*, {adj.size(-2)}, {adj.size(-1)}])")
    if adj.dim() == 2:
        edge_index = adj.nonzero().t()
        edge_attr = adj[edge_index[0], edge_index[1]]
        return edge_index, edge_attr
    else:
        flatten_adj = adj.view(-1, adj.size(-1))
        if mask is not None:
            flatten_adj = flatten_adj[mask.view(-1)]
        edge_index = flatten_adj.nonzero().t()
        edge_attr = flatten_adj[edge_index[0], edge_index[1]]

        if mask is None:
            offset = torch.arange(
                start=0,
                end=adj.size(0) * adj.size(2),
                step=adj.size(2),
                device=adj.device,
            )
            offset = offset.repeat_interleave(adj.size(1))
        else:
            count = mask.sum(dim=-1)
            offset = cumsum(count)[:-1]
            offset = offset.repeat_interleave(count)

        edge_index[1] += offset[edge_index[0]]

        return edge_index, edge_attr


def setup_wandb(cfg):
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': cfg.general.name, 'project': f'CTDGG_{cfg.dataset["name"]}', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True),
              'reinit': True, 'mode': cfg.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt')
    return cfg


class PlaceHolder:
    def __init__(self, X, E, y, t_int=None, t=None, node_mask=None):
        self.X = X
        self.E = E
        self.y = y
        self.t_int = t_int
        self.t = t
        self.node_mask = node_mask

    def device_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.to(x.device) if self.X is not None else None
        self.E = self.E.to(x.device) if self.E is not None else None
        self.y = self.y.to(x.device) if self.y is not None else None
        return self

    def mask(self, node_mask=None):
        if node_mask is None:
            assert self.node_mask is not None
            node_mask = self.node_mask
        bs, n = node_mask.shape
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1
        diag_mask = ~torch.eye(n, dtype=torch.bool,
                               device=node_mask.device).unsqueeze(0).expand(bs, -1, -1).unsqueeze(-1)  # bs, n, n, 1

        if self.X is not None:
            self.X = self.X * x_mask
        if self.E is not None:
            self.E = self.E * e_mask1 * e_mask2 * diag_mask
        return self

    def collapse(self):
        copy = self.copy()
        copy.X = torch.argmax(self.X, dim=-1)
        copy.E = torch.argmax(self.E, dim=-1)
        x_mask = self.node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1
        copy.X[self.node_mask == 0] = - 1
        copy.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        return copy

    def __repr__(self):
        return (f"X: {self.X.shape if type(self.X) == torch.Tensor else self.X} -- " +
                f"E: {self.E.shape if type(self.E) == torch.Tensor else self.E} -- " +
                f"y: {self.y.shape if type(self.y) == torch.Tensor else self.y}")

    def copy(self):
        return PlaceHolder(X=self.X, E=self.E, y=self.y, t_int=self.t_int, t=self.t,
                           node_mask=self.node_mask)

    def permutate_adj(self):
        assert self.X.shape[-1] == 1, "Adjacency matrix permutation not implemented yet for attributed graphs"
        bs = self.X.shape[0]
        n_max = self.X.shape[1]
        x_mask = self.node_mask.unsqueeze(-1)        # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1
        for i in range(bs):
            n = self.node_mask[i].sum()
            idxs = torch.arange(n_max)
            perm = torch.randperm(n)
            idxs[:n] = perm
            self.E[i] = self.E[i, idxs, :, :]
            self.E[i] = self.E[i, :, idxs, :]

        assert (self.E == self.E.transpose(1, 2)).all()
        return self
