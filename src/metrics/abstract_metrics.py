import torch
import torchmetrics
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import Metric, MeanSquaredError

from utils import PlaceHolder
from diffusion import diffusion_utils


class CrossEntropyMetric(Metric):
    full_state_update = True

    def __init__(self):
        super().__init__(sync_on_compute=False, dist_sync_on_step=False)
        self.add_state('total_ce', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """ Update state with predictions and targets.
            preds: Predictions from model   (bs * n, d) or (bs * n * n, d)
            target: Ground truth values     (bs * n, d) or (bs * n * n, d). """
        target = torch.argmax(target, dim=-1)
        output = F.cross_entropy(preds, target, reduction='sum')
        self.total_ce += output
        self.total_samples += preds.size(0)

    def compute(self):
        return self.total_ce / self.total_samples


class CTELBOMetric(Metric):
    full_state_update = True

    def __init__(self):
        super().__init__(sync_on_compute=False, dist_sync_on_step=False)
        self.add_state('total_loss', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds, target_z0, target_ztilde, qt0, rate, mask) -> None:
        """
        preds, target_z, target_ztilde : (bs, n, d) or (bs, n * n, d)
        transition : (bs, d, d)
        rate : (bs, d, d)
        """

        # First term of ELBO (regularization)
        p0t_theta = F.softmax(preds, dim=-1)          # (bs, n, d)
        reverse_rates_ztilde = diffusion_utils.get_reverse_rate_from_z(z=target_ztilde,
                                                                       p0t_theta=p0t_theta,
                                                                       qt0=qt0,
                                                                       rate=rate)

        reg_term = torch.sum(
            reverse_rates_ztilde,
            dim=(1, 2)
        )   # (b, )

        rate_to_ztilde = target_ztilde @ rate.transpose(1, 2)         # (bs, n, d)
        # Second term (signal)
        log_sig_num = torch.log(reverse_rates_ztilde + 1e-9)         # (bs, n, d)
        qztilde_z0 = torch.diagonal(target_z0 @ qt0 @ target_ztilde.transpose(1, 2), dim1=1, dim2=2)   # (bs, n)
        # Set qztilde_z0 masked values to 1. to avoid nan in division
        qztilde_z0[~mask] = 1.
        qall_z0 = target_z0 @ qt0       # (bs, n, d)

        num_sig = torch.sum(
            rate_to_ztilde * qall_z0 * log_sig_num / qztilde_z0.unsqueeze(-1),
            dim=(1, 2)
        )   # (b, )

        rate_row_sums = torch.sum(rate, dim=-1)             # (bs, d)
        base_Z_tmp = (target_ztilde @ rate_row_sums.unsqueeze(-1)).squeeze(-1)     # (bs, n)
        base_Z = torch.sum(base_Z_tmp, dim=1)

        Z_sig_norm = base_Z.unsqueeze(-1).unsqueeze(-1) - base_Z_tmp.unsqueeze(-1) + rate_row_sums.unsqueeze(-2)  # (bs, n, d)
        denom_sig = torch.sum(
            rate_to_ztilde * qall_z0 / (qztilde_z0.unsqueeze(-1) * Z_sig_norm),
            dim=(1, 2)
        ) + 1e-9

        neg_elbo = reg_term - num_sig/denom_sig     # (b, )

        self.total_loss += torch.sum(neg_elbo)
        self.total_samples += preds.size(0)

    def compute(self):
        return self.total_loss / self.total_samples


class TrainAbstractMetricsDiscrete(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, masked_pred, masked_true, log: bool):
        pass

    def reset(self):
        pass

    def log_epoch_metrics(self):
        return None, None
