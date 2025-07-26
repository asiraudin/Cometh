import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.utils import to_dense_batch
from utils import PlaceHolder


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, residual=False):
        super().__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()

        if num_layers == 1:
            # linear mapping
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for layer in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.activation = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        x_prev = x
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.activation(x)
            if x.ndim == 2:
                x = self.bns[i](x)
            elif x.ndim == 3:
                x = self.bns[i](x.transpose(2, 1)).transpose(2, 1)
            else:
                raise ValueError('invalid dimension of x')
            if self.residual and x_prev.shape == x.shape: x = x + x_prev
            x_prev = x
        x = self.lins[-1](x)
        if self.residual and x_prev.shape == x.shape:
            x = x + x_prev
        return x


class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        # input layer
        update_net = MLP(in_channels, hidden_channels, hidden_channels, 2)
        self.layers.append(GINConv(update_net))
        # hidden layers
        for i in range(n_layers - 2):
            update_net = MLP(hidden_channels, hidden_channels, hidden_channels, 2)
            self.layers.append(GINConv(update_net))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        # output layer
        update_net = MLP(hidden_channels, hidden_channels, out_channels, 2)
        self.layers.append(GINConv(update_net))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            if i != 0:
                if x.ndim == 2:
                    x = self.bns[i - 1](x)
                elif x.ndim == 3:
                    x = self.bns[i - 1](x.transpose(2, 1)).transpose(2, 1)
                else:
                    raise ValueError('invalid x dim')
            x = layer(x, edge_index)
        return x


class MaskedGINDeepSigns(nn.Module):
    """ Sign invariant neural network with sum pooling and DeepSet.
        f(v1, ..., vk) = rho(enc(v1) + enc(-v1), ..., enc(vk) + enc(-vk))
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dim_pe, rho_num_layers):
        super().__init__()
        self.enc = GIN(in_channels, hidden_channels, out_channels, num_layers)
        self.rho = MLP(out_channels, hidden_channels, dim_pe, rho_num_layers)

    def forward(self, x, edge_index, node_mask):
        x = x.transpose(0, 1)                                           # (n_nodes, k, 1) -> (k, n_nodes, 1)
        x = self.enc(x, edge_index) + self.enc(-x, edge_index)          # (k, n_nodes, out)
        x = x.transpose(0, 1)                                           # (n_nodes, k, out)

        k = x.shape[1]
        N = x.shape[0]

        n_nodes = node_mask.sum(dim=-1)
        batched_num_nodes = torch.cat([size * n_nodes.new_ones(size) for size in n_nodes])
        mask = torch.cat([torch.arange(k).unsqueeze(0) for _ in range(N)])
        mask = (mask.to(x.device) < batched_num_nodes.unsqueeze(1)).bool()
        x[~mask] = 0

        x = x.sum(dim=1)                                                # Sum over k -> (n_nodes, out)
        x = self.rho(x)                                                 # (n_nodes, out) -> (n_nodes dim_pe)
        return x


class SignNetNodeEncoder(torch.nn.Module):
    """SignNet Positional Embedding node encoder.
    https://arxiv.org/abs/2202.13013
    https://github.com/cptq/SignNet-BasisNet

    Simplified implementation of the GraphGPS version.
    https://github.com/rampasek/GraphGPS/blob/main/graphgps/encoder/signnet_pos_encoder.py

    Uses precomputated Laplacian eigen-decomposition, but instead
    of eigen-vector sign flipping + DeepSet/Transformer, computes the PE as:
    SignNetPE(v_1, ... , v_k) = \rho ( [\phi(v_i) + \rhi(−v_i)]^k_i=1 )
    where \phi is GIN network applied to k first non-trivial eigenvectors, and
    \rho is an MLP if k is a constant, but if all eigenvectors are used then
    \rho is DeepSet with sum-pooling.

    SignNetPE of size dim_pe will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with SignNetPE.

    Args:
        cfg
    """

    def __init__(self, cfg):
        super().__init__()
        self.num_vecs = cfg.num_vecs
        self.dim_pe = cfg.dim_pe

        self.sign_inv_net = MaskedGINDeepSigns(
            in_channels=1,
            hidden_channels=cfg.phi_hidden,
            out_channels=cfg.phi_out,
            num_layers=cfg.phi_layers,
            dim_pe=cfg.dim_pe,
            rho_num_layers=cfg.rho_layers
        )

    def forward(self, data: PlaceHolder, edge_index):
        bs, n = data.X.shape[0], data.X.shape[1]
        node_mask = data.node_mask                                          # (bs, n)

        k = n if n < self.num_vecs else self.num_vecs

        eigvecs = data.X[..., -k:].unsqueeze(-1)                            # (bs, n, k, 1)
        eigvecs = eigvecs.flatten(start_dim=0, end_dim=1)                   # (bs * n, k, 1)
        eigvecs = eigvecs[node_mask.flatten()]                              # (n_nodes, k, 1)

        pos_enc = self.sign_inv_net(eigvecs, edge_index, data.node_mask)    # (n_nodes, dim_pe)

        index = torch.arange(bs, device=node_mask.device).unsqueeze(-1) + 1  # (bs, 1)
        batch_index = node_mask * index                                     # (bs, n)
        batch_index = batch_index.flatten()                                 # (bs*n, )
        batch_index = batch_index[batch_index.nonzero()].squeeze() - 1      # (n_nodes,)

        pos_enc, mask = to_dense_batch(x=pos_enc, batch=batch_index)

        assert (mask == node_mask).all()
        assert pos_enc.shape == (bs, n, self.dim_pe)

        return pos_enc


class LapPENodeEncoder(torch.nn.Module):
    """Laplace Positional Embedding node encoder.

    LapPE of size dim_pe will get appended to each node feature vector.

    Args:
        cfg : encoding config dict
    """

    def __init__(self, cfg):
        super().__init__()
        self.dim_pe = cfg.dim_pe
        self.num_vecs = cfg.num_vecs
        self.rho_type = cfg.rho_type
        # Initial projection of eigenvalue and the node's eigenvector value
        self.linear_A = nn.Linear(2, self.dim_pe)

        activation = nn.ReLU
        if self.rho_type == 'Transformer':
            # Transformer model for LapPE
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim_pe,
                                                       nhead=cfg.n_heads,
                                                       batch_first=True)
            self.pe_encoder = nn.TransformerEncoder(encoder_layer,
                                                    num_layers=cfg.rho_layers)
        else:
            # DeepSet model for LapPE
            layers = []
            if cfg.rho_layers == 1:
                layers.append(activation())
            else:
                self.linear_A = nn.Linear(2, 2 * self.dim_pe)
                layers.append(activation())
                for _ in range(cfg.rho_layers - 2):
                    layers.append(nn.Linear(2 * self.dim_pe, 2 * self.dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * self.dim_pe, self.dim_pe))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)

    def forward(self, data: PlaceHolder):
        bs, n = data.X.shape[0], data.X.shape[1]
        x_mask = data.node_mask.unsqueeze(-1)                           # (bs, n, 1)

        k = n if n < self.num_vecs else self.num_vecs

        eigvecs = data.X[..., -k:]                          # (bs, n, k)
        eigvals = data.y[..., -k:].unsqueeze(1)             # (bs, 1, k)

        assert eigvecs.shape == (bs, n, k) and eigvals.shape == (bs, 1, k)

        if self.training:
            sign_flip = torch.rand_like(eigvecs)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            eigvecs = eigvecs * sign_flip

        pos_enc = torch.stack((eigvecs, eigvals.expand(-1, n, -1)), dim=-1)                   # (bs, n, k, 2)
        assert pos_enc.shape == (bs, n, k, 2), f"Incorrect size : {pos_enc.shape}"

        pos_enc = pos_enc * x_mask.unsqueeze(-1)                        # (bs, n, k, 2)
        pos_enc = self.linear_A(pos_enc)                                # (bs, n, k, dim_pe)

        # PE encoder: a Transformer or DeepSet model
        if self.rho_type == 'Transformer':
            pos_enc = pos_enc.flatten(start_dim=0, end_dim=1)           # Flatten to 3D -> (bs*n, k, dim_pe)
            tf_mask = x_mask.expand(-1, -1, k).flatten(start_dim=0, end_dim=1)              # (bs*n, k)
            assert tf_mask.shape == pos_enc.shape[:-1]
            pos_enc = self.pe_encoder(src=pos_enc, src_key_padding_mask=~tf_mask)
            pos_enc = pos_enc.reshape(bs, n, k, -1)
            # Mask Transformer nan outputs
            nan_mask = pos_enc.isnan()
            pos_enc[nan_mask] = 0.                                      # (bs, n, k, dim_pe)
        else:
            pos_enc = self.pe_encoder(pos_enc)                          # (bs, n, k, dim_pe)
            # If k == n then mask is unchanged, if k == num_vecs, truncate node_mask at k
            k_mask = data.node_mask[:, :k].unsqueeze(1).unsqueeze(-1)   # (bs, 1, k, 1)
            pos_enc = pos_enc * k_mask

        # Sum pooling
        pos_enc = torch.sum(pos_enc, 2, keepdim=False)                  # Sum over k -> (bs, n, dim_pe)
        assert pos_enc.shape == (bs, n, self.dim_pe)

        pos_enc = pos_enc * x_mask

        return pos_enc


class RandomWalkEncoder(torch.nn.Module):
    """
    Simplified version of the KernelPENodeEncoder of GraphGPS using random walks

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """
    def __init__(self, cfg):
        super().__init__()
        self.dim_pe = cfg.dim_pe
        self.num_rw_steps = len(cfg.n_steps)
        self.raw_norm_node = nn.BatchNorm1d(self.num_rw_steps)
        self.raw_norm_edge = nn.BatchNorm1d(self.dim_pe)

        self.pe_node_encoder = nn.Linear(self.num_rw_steps, self.dim_pe)
        self.pe_edge_encoder = nn.Linear(self.num_rw_steps+1, self.dim_pe)

    def forward(self, data):
        x_mask = data.node_mask.unsqueeze(-1)                           # (bs, n, 1)
        bs, n = x_mask.shape[:2]

        pos_enc_X = data.X[..., -self.num_rw_steps:]                      # (bs, n, num_steps)
        pos_enc_X = self.raw_norm_node(pos_enc_X.transpose(1, 2)).transpose(1, 2)
        pos_enc_X = self.pe_node_encoder(pos_enc_X)  # (bs, n, dim_pe)
        pos_enc_X = pos_enc_X * x_mask

        pos_enc_E = data.E[..., -self.num_rw_steps-1:]
        pos_enc_E = self.pe_edge_encoder(pos_enc_E)             # (bs, n, n, dim_pe)
        pos_enc_E = pos_enc_E.flatten(start_dim=1, end_dim=2)   # (bs, n*n, dim_pe)
        pos_enc_E = self.raw_norm_edge(pos_enc_E.transpose(1, 2)).transpose(1, 2)
        pos_enc_E = pos_enc_E.reshape(bs, n, n, -1)             # (bs, n, n, dim_pe)

        assert pos_enc_E.shape == (bs, n, n, self.dim_pe)
        pos_enc_E = pos_enc_E * x_mask.unsqueeze(1) * x_mask.unsqueeze(2)

        return pos_enc_X, pos_enc_E
