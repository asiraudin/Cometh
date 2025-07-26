import torch
import torch.nn as nn
import time
import wandb

from metrics.abstract_metrics import CTELBOMetric, CrossEntropyMetric


class TrainLoss(nn.Module):
    """ Train with Cross entropy"""
    def __init__(self, train, lambda_train):
        super().__init__()

        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()

        self.lambda_train = lambda_train

    def forward(self, preds, z_0, log: bool):
        node_mask = z_0.node_mask
        bs, n = node_mask.shape

        diag_mask = ~torch.eye(n, device=node_mask.device, dtype=torch.bool).unsqueeze(0).repeat(bs, 1, 1)
        edge_mask = diag_mask & node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)

        masked_pred_X = preds.X[node_mask]
        masked_pred_E = preds.E[edge_mask]

        masked_X0 = z_0.X[node_mask]
        masked_E0 = z_0.E[edge_mask]

        # Check that the masking is correct
        assert (masked_X0 != 0.).any(dim=-1).all()
        assert (masked_E0 != 0.).any(dim=-1).all()

        loss_X = self.node_loss(masked_pred_X, masked_X0) if masked_X0.numel() > 0 else 0.0
        loss_E = self.edge_loss(masked_pred_E, masked_E0) if masked_E0.numel() > 0 else 0.0

        batch_loss = (self.lambda_train[0] * loss_X +
                      + self.lambda_train[1] * loss_E)
        to_log = {"train_loss/X_loss": self.lambda_train[0] * self.node_loss.compute() if z_0.X.numel() > 0 else -1.0,
                  "train_loss/E_loss": self.lambda_train[1] * self.edge_loss.compute() if z_0.E.numel() > 0 else -1.0,
                  "train_loss/batch_loss": batch_loss.detach()} if log else None

        if log and wandb.run:
            wandb.log(to_log, commit=True)
        return batch_loss, to_log

    def reset(self):
        for metric in [self.node_loss, self.edge_loss]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_loss = self.node_loss.compute().detach() if self.node_loss.total_samples > 0 else -1.0
        epoch_edge_loss = self.edge_loss.compute().detach() if self.edge_loss.total_samples > 0 else -1.0

        key = "train_epoch"
        to_log = {f"{key}/X_loss": epoch_node_loss,
                  f"{key}/E_loss": epoch_edge_loss}

        if wandb.run:
            wandb.log(to_log, commit=False)
        return to_log


class ValidationLoss(nn.Module):
    """ Compute the ELBO for validation"""
    def __init__(self):
        super().__init__()
        self.node_loss = CTELBOMetric()
        self.edge_loss = CTELBOMetric()

    def forward(self, preds, z_0, z_tilde, qt0, rate):
        node_mask = z_0.node_mask
        bs, n = node_mask.shape
        x_classes = z_0.X.size(-1)

        diag_mask = ~torch.eye(n, device=node_mask.device, dtype=torch.bool).unsqueeze(0).repeat(bs, 1, 1)
        edge_mask = diag_mask & node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)

        loss_X = self.node_loss(preds=preds.X, target_z0=z_0.X, target_ztilde=z_tilde.X,
                                qt0=qt0.X, rate=rate.X, mask=node_mask) if z_0.X.numel() > 0 and x_classes > 1 else 0.0

        loss_E = self.edge_loss(preds=preds.E.flatten(start_dim=1, end_dim=2),
                                target_z0=z_0.E.flatten(start_dim=1, end_dim=2),
                                target_ztilde=z_tilde.E.flatten(start_dim=1, end_dim=2),
                                qt0=qt0.E, rate=rate.E, mask=edge_mask.flatten(start_dim=1, end_dim=2)) \
            if z_0.E.numel() > 0 else 0.0

        batch_loss = loss_X + loss_E

        return batch_loss

    def reset(self):
        for metric in [self.node_loss, self.edge_loss]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_loss = self.node_loss.compute().item() if self.node_loss.total_samples > 0 else -1.0
        epoch_edge_loss = self.edge_loss.compute().item() if self.edge_loss.total_samples > 0 else -1.0

        to_log = {"val_epoch/X_elbo": epoch_node_loss,
                  "val_epoch/E_elbo": epoch_edge_loss,
                  "val_epoch/elbo": epoch_edge_loss + epoch_node_loss}

        if wandb.run:
            wandb.log(to_log, commit=False)
        return to_log
