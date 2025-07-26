import torch
from torch.nn import functional as F
from utils import PlaceHolder
from torch.distributions.poisson import Poisson
import numpy as np
import warnings
import torch

warnings.filterwarnings("ignore", category=UserWarning)


def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(dim=-1)


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask.long())).abs().max().detach() < 1e-4, \
        'Variables not masked properly.'


def get_upper_triangular_mask(edge_matrix):
    """
    Get upper triangular part of edge noise, without main diagonal
    """
    upper_triangular_mask = torch.zeros_like(edge_matrix)
    indices = torch.triu_indices(row=edge_matrix.size(1), col=edge_matrix.size(2), offset=1)
    if len(edge_matrix.shape) == 4:
        upper_triangular_mask[:, indices[0], indices[1], :] = 1
    else:
        upper_triangular_mask[:, indices[0], indices[1]] = 1

    return upper_triangular_mask


def sample_discrete_features(probX, probE, node_mask):
    ''' Sample features from multinomial distribution with given probabilities (probX, probE, proby)
        :param probX: bs, n, dx_out        node features
        :param probE: bs, n, n, de_out     edge features
        :param proby: bs, dy_out           global features.
    '''
    bs, n = node_mask.shape
    # Noise X
    # The masked rows should define probability distributions as well
    probX[~node_mask] = 1 / probX.shape[-1]

    # Flatten the probability tensor to sample with multinomial
    probX = probX.reshape(bs * n, -1)       # (bs * n, dx_out)

    # Sample X
    X_t = probX.multinomial(1)                                  # (bs * n, 1)
    X_t = X_t.reshape(bs, n)     # (bs, n)

    # Noise E
    # The masked rows should define probability distributions as well
    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diag_mask = torch.eye(n).unsqueeze(0).expand(bs, -1, -1)

    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask.bool()] = 1 / probE.shape[-1]

    probE = probE.reshape(bs * n * n, -1)    # (bs * n * n, de_out)

    # Sample E
    E_t = probE.multinomial(1).reshape(bs, n, n)   # (bs, n, n)
    E_t = torch.triu(E_t, diagonal=1)
    E_t = (E_t + torch.transpose(E_t, 1, 2))

    # return PlaceHolder(X=X_t, charges=charges_t, E=E_t, y=torch.zeros(bs, 0).type_as(X_t))
    return PlaceHolder(X=X_t, E=E_t, y=torch.zeros(bs, 0).type_as(X_t))


def sample_transition_dims(z, rate, node_mask):
    bs, n_max = node_mask.shape

    float_mask = node_mask.float()  # (bs, n)
    edge_mask = float_mask.unsqueeze(1) * float_mask.unsqueeze(-1)  # (bs, n, n)

    upper_triangular_mask = get_upper_triangular_mask(edge_mask)

    edge_mask = edge_mask * upper_triangular_mask
    dimension_mask = torch.cat((float_mask, edge_mask.flatten(start_dim=1)), dim=1)  # (bs, n + n**2)

    rate_out_X = z.X.float() @ rate.X                                                       # (bs, n, dx)
    rate_out_E = z.E.flatten(start_dim=1, end_dim=2).float() @ rate.E                       # (bs, n * n, de)

    rate_out_X_sum = torch.sum(rate_out_X, dim=-1)                                  # (bs, n)
    rate_out_E_sum = torch.sum(rate_out_E, dim=-1)                                  # (bs, n*n)

    rate_out_sum = torch.cat((rate_out_X_sum, rate_out_E_sum), dim=1)      # (bs, n + n*n)
    masked_rate_out_sum = rate_out_sum * dimension_mask

    # Sample a transition dimension (node or edge) for each sample
    transition_dim = masked_rate_out_sum.multinomial(1)                                          # (bs, 1)

    node_transitions_idx = torch.where(transition_dim < n_max)[0]  # (number of node transitions,)
    edge_transition_idx = torch.where(transition_dim >= n_max)[0]  # same for edges

    node_transitions = torch.cat((node_transitions_idx.unsqueeze(-1), transition_dim[node_transitions_idx]), dim=1)
    edge_transitions = torch.cat((edge_transition_idx.unsqueeze(-1), transition_dim[edge_transition_idx] - n_max), dim=1)

    transitions = PlaceHolder(X=node_transitions, E=edge_transitions, y=None)
    return transitions


def sample_auxiliary_features(z, rate, transitions):
    Xt_transition = z.X[transitions.X[:, 0], transitions.X[:, 1], :].float()  # (n of node transitions, dx)
    Et_transition = z.E.flatten(start_dim=1, end_dim=2)[transitions.E[:, 0], transitions.E[:, 1], :].float()  # same for edges

    prob_Xtilde = Xt_transition.unsqueeze(1) @ rate.X[transitions.X[:, 0]]
    prob_Etilde = Et_transition.unsqueeze(1) @ rate.E[transitions.E[:, 0]]

    sampled_Xtilde = prob_Xtilde.squeeze().multinomial(1).squeeze() \
        if z.X.size(-1) > 1 else torch.tensor((0, 1), device=z.X.device)
    sampled_Etilde = prob_Etilde.squeeze().multinomial(1).squeeze()

    sampled_ztilde = PlaceHolder(X=sampled_Xtilde, E=sampled_Etilde, y=None)
    return sampled_ztilde


def get_reverse_rate_from_z(z, p0t_theta, qt0, rate):
    qz_all = z @ qt0.transpose(1, 2) + 1e-9
    sum_over_z0 = (p0t_theta.unsqueeze(2) @ (qt0.unsqueeze(1) / qz_all.unsqueeze(-1))).squeeze(dim=2)

    rate_to_z = z @ rate.transpose(1, 2)
    return rate_to_z * sum_over_z0


def get_corrector_rate(z, rate, reverse_rate):
    """
    z : (bs, n, d)
    rate : (bs, d, d)
    reverse : (bs, n, d)
    """
    rate_from_z = z @ rate
    return rate_from_z + reverse_rate


def leap(z, reverse_rate, tau):
    """Sample transitions from Poisson distribution. Reject multiple transitions in one dimension.
    """
    poisson_dist = Poisson(tau * reverse_rate)
    jumps = poisson_dist.sample()                   # (bs, n, d)
    jump_dim_sum = torch.sum(jumps, dim=-1)         # (bs, n)
    jump_mask = jump_dim_sum == 1

    z[jump_mask] = jumps[jump_mask]
    return z


def process_rate(rate, target):
    # Set diag to 0
    rate = rate - torch.diag(torch.diag(rate))
    # Set each row sum to 0
    rate = rate - torch.diag(torch.sum(rate, dim=1))
    # print("process rate: rate",rate)
    if target == "uniform":
        eigvals, eigvecs = torch.linalg.eigh(rate)
        inveigvecs = torch.linalg.inv(eigvecs)
    elif target == "marginal":
        eigvals, eigvecs = torch.linalg.eig(rate)
        # print("eigvals", eigvals, "eigvecs", eigvecs)
        eigvals, eigvecs = eigvals.float(), eigvecs.float()
        inveigvecs = torch.linalg.inv(eigvecs)
    else:
        raise NotImplementedError("This target distribution is not implemented.")

    return rate, eigvecs, inveigvecs, eigvals


def cosine_beta_schedule_discrete(timesteps, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas
    return betas.squeeze()


def custom_beta_schedule_discrete(timesteps, average_num_nodes=50, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas

    assert timesteps >= 100

    p = 4 / 5       # 1 - 1 / num_edge_classes
    num_edges = average_num_nodes * (average_num_nodes - 1) / 2

    # First 100 steps: only a few updates per graph
    updates_per_graph = 1.2
    beta_first = updates_per_graph / (p * num_edges)

    betas[betas < beta_first] = beta_first
    return np.array(betas)

