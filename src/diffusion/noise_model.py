import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math

import utils
from diffusion import diffusion_utils

import hydra
import omegaconf
import matplotlib.pyplot as plt

from abc import abstractmethod


class NoiseModel:
    def __init__(self, cfg):
        # Define the transition matrices for the discrete features
        self.X_classes = None
        self.E_classes = None
        self.y_classes = None

        self.X_marginals = None
        self.E_marginals = None
        self.y_marginals = None

        self.X_rate = None
        self.E_rate = None
        self.y_rate = None

        self.eigvals = None
        self.eigvecs = None

        self.T = cfg.model.diffusion_steps
        self.rate_const = cfg.model.rate_constant
        self.min_time = cfg.model.min_time

        ts, tau = np.linspace(self.min_time, 1.0, self.T, retstep=True)
        self.tau = tau
        self.ts = torch.from_numpy(ts)
        self.corrector_tau_multiplier = cfg.model.corrector_tau_multiplier


    @abstractmethod
    def rate(self, t_float, include_diag=False):
        pass

    @abstractmethod
    def transition(self, t_float):
        pass

    def get_limit_dist(self):
        X_marginals = self.X_marginals + 1e-7
        X_marginals = X_marginals / torch.sum(X_marginals)
        E_marginals = self.E_marginals + 1e-7
        E_marginals = E_marginals / torch.sum(E_marginals)

        limit_dist = utils.PlaceHolder(X=X_marginals, E=E_marginals, y=None)
        return limit_dist

    def sample_limit_dist(self, node_mask):
        """ Sample from the limit distribution of the diffusion process"""
        bs, n_max = node_mask.shape
        x_limit = self.X_marginals.expand(bs, n_max, -1)
        e_limit = self.E_marginals[None, None, None, :].expand(bs, n_max, n_max, -1)

        U_X = x_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max).to(node_mask.device)
        U_E = e_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max, n_max).to(node_mask.device)
        U_y = torch.zeros((bs, 0), device=node_mask.device)

        U_X = F.one_hot(U_X, num_classes=x_limit.shape[-1]).float()
        U_E = F.one_hot(U_E, num_classes=e_limit.shape[-1]).float()

        # Get upper triangular part of edge noise, without main diagonal
        upper_triangular_mask = torch.zeros_like(U_E)
        indices = torch.triu_indices(row=U_E.size(1), col=U_E.size(2), offset=1)
        upper_triangular_mask[:, indices[0], indices[1], :] = 1

        U_E = U_E * upper_triangular_mask
        U_E = (U_E + torch.transpose(U_E, 1, 2))
        assert (U_E == torch.transpose(U_E, 1, 2)).all()

        t_array = U_X.new_ones((U_X.shape[0], 1))
        t_int_array = self.T * t_array.long()
        return utils.PlaceHolder(X=U_X, E=U_E, y=U_y, t_int=t_int_array, t=t_array,
                                 node_mask=node_mask).mask(node_mask)

    def sample_zs_from_zt_and_pred(self, z_t, preds, last_pass, corrector=False):
        bs, n = z_t.node_mask.shape
        node_mask = z_t.node_mask
        t_float = z_t.t

        p0t_theta_X = F.softmax(preds.X, dim=-1)        # (bs, n, d)
        p0t_theta_E = F.softmax(preds.E, dim=-1)        # (bs, n, n, d)

        if last_pass:
            X0max = torch.max(p0t_theta_X, dim=-1)[1]
            E0max = torch.max(p0t_theta_E, dim=-1)[1]

            X_s = F.one_hot(X0max, num_classes=p0t_theta_X.size(-1))
            E_s = F.one_hot(E0max, num_classes=p0t_theta_E.size(-1))

            z_s = utils.PlaceHolder(X=X_s,
                                    E=E_s, y=torch.zeros(z_t.y.shape[0], 0, device=X_s.device),
                                    t_int=z_t.t_int, t=torch.zeros_like(z_t.t),
                                    node_mask=node_mask).mask().device_as(t_float)
            return z_s

        qt0 = self.transition(t_float)
        rate = self.rate(t_float)

        reverse_rates_X = diffusion_utils.get_reverse_rate_from_z(z=z_t.X, p0t_theta=p0t_theta_X,
                                                                  qt0=qt0.X, rate=rate.X)           # (bs, n, dx)

        reverse_rates_E = diffusion_utils.get_reverse_rate_from_z(z=z_t.E.flatten(start_dim=1, end_dim=2),
                                                                  p0t_theta=p0t_theta_E.flatten(start_dim=1, end_dim=2),
                                                                  qt0=qt0.E,
                                                                  rate=rate.E)     # (bs, n*n, de)
        if not corrector:
            X_s = diffusion_utils.leap(z=z_t.X, reverse_rate=reverse_rates_X, tau=self.tau)                    # (bs, n, d)
            E_s = diffusion_utils.leap(z=z_t.E.flatten(start_dim=1, end_dim=2), reverse_rate=reverse_rates_E, tau=self.tau)      # (bs, n*n, d)
        else:
            X_t = z_t.X
            E_t = z_t.E.flatten(start_dim=1, end_dim=2)

            corrector_rates_X = diffusion_utils.get_corrector_rate(X_t, rate.X, reverse_rates_X)
            corrector_rates_E = diffusion_utils.get_corrector_rate(E_t, rate.E, reverse_rates_E)

            X_s = diffusion_utils.leap(z=X_t, reverse_rate=corrector_rates_X, tau=self.tau*self.corrector_tau_multiplier)
            E_s = diffusion_utils.leap(z=E_t, reverse_rate=corrector_rates_E, tau=self.tau*self.corrector_tau_multiplier)

        E_s = E_s.reshape(bs, n, n, -1)
        upper_triangular_mask = diffusion_utils.get_upper_triangular_mask(E_s)

        E_s = E_s * upper_triangular_mask
        E_s = (E_s + torch.transpose(E_s, 1, 2))

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (z_t.X.shape == X_s.shape) and (z_t.E.shape == E_s.shape)

        self.ts = self.ts.to(X_s.device)
        s_int = (z_t.t_int - 1).to(X_s.device) if not corrector else z_t.t_int
        if corrector:
            s = t_float
        else:
            s = self.ts[s_int - 1].float() if s_int[0] > 0 else self.ts[torch.zeros_like(s_int)].float()

        z_s = utils.PlaceHolder(X=X_s,
                                E=E_s, y=torch.zeros(z_t.y.shape[0], 0, device=X_s.device),
                                t_int=s_int, t=s,
                                node_mask=node_mask).mask()

        return z_s

    def apply_noise(self, dense_data, validation=False):
        device = dense_data.X.device
        t_float = torch.rand((dense_data.X.size(0), 1), device=device)
        t_float = t_float * (1 - self.min_time) + self.min_time

        assert (t_float < 1.).all()
        assert (t_float >= self.min_time).all()

        qt0 = self.transition(t_float)
        qt0_Xt = dense_data.X @ qt0.X                          # (bs, n, dx)
        qt0_Et = dense_data.E @ qt0.E.unsqueeze(1)             # (bs, n, n, de)

        sampled_t = diffusion_utils.sample_discrete_features(qt0_Xt, qt0_Et, dense_data.node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.X_classes)        # (bs, n, dx)
        E_t = F.one_hot(sampled_t.E, num_classes=self.E_classes)        # (bs, n, n, de)
        assert (dense_data.X.shape == X_t.shape) and (dense_data.E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=dense_data.y,
                                t=t_float, node_mask=dense_data.node_mask).mask()
        if validation:
            # To compute the likelihood, we need to the full ELBO which requires to sample an auxiliary z_tilde

            # sample x_tilde
            bs, n_max = dense_data.node_mask.shape
            rate = self.rate(t_float)

            transitions = diffusion_utils.sample_transition_dims(z=z_t, rate=rate, node_mask=dense_data.node_mask)
            sampled_ztilde = diffusion_utils.sample_auxiliary_features(z=z_t, rate=rate, transitions=transitions)

            X_tilde = z_t.X.clone()
            E_tilde = z_t.E.flatten(start_dim=1, end_dim=2).clone()
            if self.X_classes > 1:
                X_tilde[transitions.X[:, 0], transitions.X[:, 1], :] = F.one_hot(sampled_ztilde.X, num_classes=self.X_classes)
            E_tilde[transitions.E[:, 0], transitions.E[:, 1], :] = F.one_hot(sampled_ztilde.E, num_classes=self.E_classes)
            E_tilde = E_tilde.view(bs, n_max, n_max, -1)

            upper_triangular_mask = diffusion_utils.get_upper_triangular_mask(E_tilde)

            E_tilde = E_tilde * upper_triangular_mask
            E_tilde = (E_tilde + torch.transpose(E_tilde, 1, 2))

            assert (E_tilde == torch.transpose(E_tilde, 1, 2)).all()
            assert (E_tilde.size() == z_t.E.size() and X_tilde.size() == z_t.X.size())
            assert (E_tilde.size() == z_t.E.size() and X_tilde.size() == z_t.X.size())

            z_tilde = utils.PlaceHolder(X=X_tilde.float(), E=E_tilde.float(), y=dense_data.y,
                                        t=t_float, node_mask=dense_data.node_mask).mask()
            return z_t, z_tilde, qt0, rate

        return z_t


class MixinConstant:
    def rate(self, t_float, include_diag=False):
        bs = t_float.shape[0]
        r_x = torch.tile(self.X_rate.view(1, self.X_classes, self.X_classes), (bs, 1, 1))
        r_e = torch.tile(self.E_rate.view(1, self.E_classes, self.E_classes), (bs, 1, 1))
        r_y = torch.tile(self.y_rate.view(1, self.y_classes, self.y_classes), (bs, 1, 1))

        if not include_diag:
            r_x = r_x - torch.diag_embed(torch.diagonal(r_x, dim1=-2, dim2=-1), dim1=-2, dim2=-1)
            r_e = r_e - torch.diag_embed(torch.diagonal(r_e, dim1=-2, dim2=-1), dim1=-2, dim2=-1)
            r_y = r_y - torch.diag_embed(torch.diagonal(r_y, dim1=-2, dim2=-1), dim1=-2, dim2=-1)

        return utils.PlaceHolder(X=r_x, E=r_e, y=r_y).device_as(t_float)

    def transition(self, t_float):
        bs = t_float.shape[0]
        eigvals = self.eigvals.device_as(t_float)
        eigvecs = self.eigvecs.device_as(t_float)
        inveigvecs = self.inveigvecs.device_as(t_float)

        X_transitions = eigvecs.X.view(1, self.X_classes, self.X_classes) @ \
                        torch.diag_embed(torch.exp(eigvals.X.view(1, self.X_classes) * t_float.view(bs, 1))) @ \
                        inveigvecs.X.view(1, self.X_classes, self.X_classes)

        E_transitions = eigvecs.E.view(1, self.E_classes, self.E_classes) @ \
                        torch.diag_embed(torch.exp(eigvals.E.view(1, self.E_classes) * t_float.view(bs, 1))) @ \
                        inveigvecs.E.view(1, self.E_classes, self.E_classes)

        y_transitions = eigvecs.y.view(1, self.y_classes, self.y_classes) @ \
                        torch.diag_embed(torch.exp(eigvals.y.view(1, self.y_classes) * t_float.view(bs, 1))) @ \
                        inveigvecs.y.view(1, self.y_classes, self.y_classes)

        # if torch.min(transitions) < -1e-6:
        #     print(f"[Warning] UniformRate, large negative transition values {torch.min(transitions)}")

        X_transitions[X_transitions < 1e-8] = 0.0
        E_transitions[E_transitions < 1e-8] = 0.0
        y_transitions[y_transitions < 1e-8] = 0.0

        return utils.PlaceHolder(X=X_transitions,
                                 E=E_transitions,
                                 y=y_transitions).device_as(t_float)


class MixinCosine:
    @staticmethod
    def _integral_rate_scalar(t_float):
        return 1 - torch.cos(t_float * math.pi/2)

    @staticmethod
    def _rate_scalar(t_float):
        return (math.pi / 2) * torch.sin(t_float * math.pi/2)

    def rate(self, t_float, include_diag=False):
        bs = t_float.shape[0]
        rate_scalars = self._rate_scalar(t_float).to(self.X_rate.device)
        r_x = self.X_rate.view(1, self.X_classes, self.X_classes) * rate_scalars.view(bs, 1, 1)
        r_e = self.E_rate.view(1, self.E_classes, self.E_classes) * rate_scalars.view(bs, 1, 1)
        r_y = self.y_rate.view(1, self.y_classes, self.y_classes) * rate_scalars.view(bs, 1, 1)

        if not include_diag:
            r_x = r_x - torch.diag_embed(torch.diagonal(r_x, dim1=-2, dim2=-1), dim1=-2, dim2=-1)
            r_e = r_e - torch.diag_embed(torch.diagonal(r_e, dim1=-2, dim2=-1), dim1=-2, dim2=-1)
            r_y = r_y - torch.diag_embed(torch.diagonal(r_y, dim1=-2, dim2=-1), dim1=-2, dim2=-1)

        return utils.PlaceHolder(X=r_x, E=r_e, y=r_y).device_as(t_float)

    def transition(self, t_float):
        bs = t_float.shape[0]
        eigvals = self.eigvals.device_as(t_float)
        eigvecs = self.eigvecs.device_as(t_float)
        inveigvecs = self.inveigvecs.device_as(t_float)

        integral_rate_scalars = self._integral_rate_scalar(t_float)
        beta_X_eigvals = integral_rate_scalars.view(bs, 1) * eigvals.X.view(1, self.X_classes)
        beta_E_eigvals = integral_rate_scalars.view(bs, 1) * eigvals.E.view(1, self.E_classes)
        beta_y_eigvals = integral_rate_scalars.view(bs, 1) * eigvals.y.view(1, self.y_classes)

        X_transitions = eigvecs.X.view(1, self.X_classes, self.X_classes) @ \
                        torch.diag_embed(torch.exp(beta_X_eigvals)) @ \
                        inveigvecs.X.view(1, self.X_classes, self.X_classes)

        E_transitions = eigvecs.E.view(1, self.E_classes, self.E_classes) @ \
                        torch.diag_embed(torch.exp(beta_E_eigvals)) @ \
                        inveigvecs.E.view(1, self.E_classes, self.E_classes)

        y_transitions = eigvecs.y.view(1, self.y_classes, self.y_classes) @ \
                        torch.diag_embed(torch.exp(beta_y_eigvals)) @ \
                        inveigvecs.y.view(1, self.y_classes, self.y_classes)

        # if torch.min(transitions) < -1e-6:
        #     print(f"[Warning] UniformRate, large negative transition values {torch.min(transitions)}")

        X_transitions[X_transitions < 1e-8] = 0.0
        E_transitions[E_transitions < 1e-8] = 0.0
        y_transitions[y_transitions < 1e-8] = 0.0

        return utils.PlaceHolder(X=X_transitions,
                                 E=E_transitions,
                                 y=y_transitions).device_as(t_float)


class UniformRate(NoiseModel):
    def __init__(self, cfg, output_dims):
        super().__init__(cfg)
        self.X_classes = output_dims.X
        self.E_classes = output_dims.E
        self.y_classes = output_dims.y

        self.X_marginals = torch.ones(self.X_classes) / self.X_classes
        self.E_marginals = torch.ones(self.E_classes) / self.E_classes
        self.y_marginals = torch.ones(self.y_classes) / self.y_classes

        X_rate = cfg.model.rate_constant[0] * torch.ones((self.X_classes, self.X_classes))
        E_rate = cfg.model.rate_constant[1] * torch.ones((self.E_classes, self.E_classes))
        y_rate = cfg.model.rate_constant[2] * torch.ones((self.y_classes, self.y_classes))

        self.X_rate, X_eigvecs, X_inveigvecs, X_eigvals = diffusion_utils.process_rate(X_rate, "uniform")
        self.E_rate, E_eigvecs, E_inveigvecs, E_eigvals = diffusion_utils.process_rate(E_rate, "uniform")
        self.y_rate, y_eigvecs, y_inveigvecs, y_eigvals = diffusion_utils.process_rate(y_rate, "uniform")

        self.eigvals = utils.PlaceHolder(X=X_eigvals, E=E_eigvals, y=y_eigvals)
        self.eigvecs = utils.PlaceHolder(X=X_eigvecs, E=E_eigvecs, y=y_eigvecs)
        self.inveigvecs = utils.PlaceHolder(X=X_inveigvecs, E=E_inveigvecs, y=y_inveigvecs)

    @abstractmethod
    def rate(self, t_float, include_diag=False):
        pass

    @abstractmethod
    def transition(self, t_float):
        pass


class AbsorbingRate(NoiseModel):
    def __init__(self, cfg, x_marginals, e_marginals, y_classes):
        super().__init__(cfg)
        self.X_classes = len(x_marginals)
        self.E_classes = len(e_marginals)
        self.y_classes = y_classes

        x_absorbing = F.one_hot(x_marginals.argmax(), num_classes=self.X_classes)
        e_absorbing = F.one_hot(e_marginals.argmax(), num_classes=self.E_classes)

        self.X_marginals = x_absorbing
        self.E_marginals = e_absorbing
        self.y_marginals = torch.ones(self.y_classes) / self.y_classes

        X_rate = cfg.model.rate_constant[0] * x_absorbing.unsqueeze(0).expand(self.X_classes, -1)
        E_rate = cfg.model.rate_constant[1] * e_absorbing.unsqueeze(0).expand(self.E_classes, -1)
        y_rate = cfg.model.rate_constant[2] * torch.ones((self.y_classes, self.y_classes))

        self.X_rate, X_eigvecs, X_inveigvecs, X_eigvals = diffusion_utils.process_rate(X_rate, "marginal")
        self.E_rate, E_eigvecs, E_inveigvecs, E_eigvals = diffusion_utils.process_rate(E_rate, "marginal")
        self.y_rate, y_eigvecs, y_inveigvecs, y_eigvals = diffusion_utils.process_rate(y_rate, "marginal")

        self.eigvals = utils.PlaceHolder(X=X_eigvals, E=E_eigvals, y=y_eigvals)
        self.eigvecs = utils.PlaceHolder(X=X_eigvecs, E=E_eigvecs, y=y_eigvecs)
        self.inveigvecs = utils.PlaceHolder(X=X_inveigvecs, E=E_inveigvecs, y=y_inveigvecs)

    @abstractmethod
    def rate(self, t_float, include_diag=False):
        pass

    @abstractmethod
    def transition(self, t_float):
        pass


class MarginalRate(NoiseModel):
    def __init__(self, cfg, x_marginals, e_marginals, y_classes):
        super().__init__(cfg)
        self.X_classes = len(x_marginals)
        self.E_classes = len(e_marginals)
        self.y_classes = y_classes

        self.X_marginals = x_marginals
        self.E_marginals = e_marginals
        self.y_marginals = torch.ones(self.y_classes) / self.y_classes

        X_rate = cfg.model.rate_constant[0] * x_marginals.unsqueeze(0).expand(self.X_classes, -1)
        E_rate = cfg.model.rate_constant[1] * e_marginals.unsqueeze(0).expand(self.E_classes, -1)
        y_rate = cfg.model.rate_constant[2] * torch.ones((self.y_classes, self.y_classes))

        self.X_rate, X_eigvecs, X_inveigvecs, X_eigvals = diffusion_utils.process_rate(X_rate, "marginal")
        self.E_rate, E_eigvecs, E_inveigvecs, E_eigvals = diffusion_utils.process_rate(E_rate, "marginal")
        self.y_rate, y_eigvecs, y_inveigvecs, y_eigvals = diffusion_utils.process_rate(y_rate, "marginal")

        self.eigvals = utils.PlaceHolder(X=X_eigvals, E=E_eigvals, y=y_eigvals)
        self.eigvecs = utils.PlaceHolder(X=X_eigvecs, E=E_eigvecs, y=y_eigvecs)
        self.inveigvecs = utils.PlaceHolder(X=X_inveigvecs, E=E_inveigvecs, y=y_inveigvecs)

    @abstractmethod
    def rate(self, t_float, include_diag=False):
        pass

    @abstractmethod
    def transition(self, t_float):
        pass


class UniformRateConstant(MixinConstant, UniformRate):
    def __init__(self, cfg, output_dims):
        super().__init__(cfg, output_dims)


class UniformRateCosine(MixinCosine, UniformRate):
    def __init__(self, cfg, output_dims):
        super().__init__(cfg, output_dims)


class AbsorbingRateConstant(MixinConstant, AbsorbingRate):
    def __init__(self, cfg, x_marginals, e_marginals, y_classes):
        super().__init__(cfg, x_marginals, e_marginals, y_classes)


class AbsorbingRateCosine(MixinCosine, AbsorbingRate):
    def __init__(self, cfg, x_marginals, e_marginals, y_classes):
        super().__init__(cfg, x_marginals, e_marginals, y_classes)


class MarginalRateConstant(MixinConstant, MarginalRate):
    def __init__(self, cfg, x_marginals, e_marginals, y_classes):
        super().__init__(cfg, x_marginals, e_marginals, y_classes)


class MarginalRateCosine(MixinCosine, MarginalRate):
    def __init__(self, cfg, x_marginals, e_marginals, y_classes):
        super().__init__(cfg, x_marginals, e_marginals, y_classes)


if __name__ == '__main__':
    @hydra.main(version_base='1.3.2', config_path='../../configs', config_name='config')
    def main(cfg: omegaconf.DictConfig):
        x_marginals = torch.tensor([0.0, 0.72200687, 0.1364436, 0.10383305, 0.01433876,
                                    0.01637907, 0.00546271, 0.00153594])
        e_marginals = torch.tensor([8.97329175e-01, 4.73992797e-02, 6.30609990e-03, 3.54786448e-04, 4.86106585e-02])
        noise_model_mar = MarginalRateCosine(cfg, x_marginals=x_marginals, e_marginals=e_marginals, y_classes=0)

        ts = noise_model_mar.ts.float()
        full_atom_encoder = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'S': 5, 'Cl': 6, 'Br': 7}
        decoder_nodes = list(full_atom_encoder.keys())
        decoder_edges = ["No bond", "Single", "Double", "Triple", "Aromatic"]
        transitions_X = noise_model_mar.transition(ts).X
        transitions_E = noise_model_mar.transition(ts).E

        # for i, atom in enumerate(decoder):
        #     transition = transitions_X[:, i, i]
        #     # alpha_bar = (transition - x_marginals[i]) / (1 - x_marginals[i])
        #     # plt.plot(ts, alpha_bar)
        #     plt.plot(ts, transition)
        # plt.legend(decoder)
        for i, atom in enumerate(decoder_edges):
            transition = transitions_E[:, i, i]
            # alpha_bar = (transition - x_marginals[i]) / (1 - x_marginals[i])
            # plt.plot(ts, alpha_bar)
            plt.plot(ts, transition)
        plt.legend(decoder_edges)
        plt.show()
    main()
