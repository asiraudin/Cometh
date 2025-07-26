import torch
import utils
import time
from models.abstract_diffusion_model import AbstractDiffusionModel
import wandb

from contextlib import nullcontext


class DiffusionModel(AbstractDiffusionModel):
    def __init__(self, cfg, dataset_infos, train_metrics, val_sampling_metrics, test_sampling_metrics,
                 visualization_tools, extra_features, domain_features):
        super().__init__(cfg, dataset_infos, train_metrics, val_sampling_metrics, test_sampling_metrics,
                         visualization_tools, extra_features, domain_features)

    def training_step(self, data, i):
        if data.edge_index.numel() == 0:
            print("Found a batch with no edges. Skipping.")
            return
        dense_data = utils.to_dense(data, self.dataset_infos.num_node_types, self.dataset_infos.num_edge_types)
        z_t = self.noise_model.apply_noise(dense_data)
        extra_data = self.compute_extra_data(z_t)

        pred = self.forward(z_t, extra_data)
        loss, tl_log_dict = self.train_loss(preds=pred, z_0=dense_data,
                                            log=i % self.log_every_steps == 0)

        tm_log_dict = self.train_metrics(masked_pred=pred, masked_true=dense_data,
                                         log=i % self.log_every_steps == 0)

        return loss

    def validation_step(self, data, i):
        dense_data = utils.to_dense(data, self.dataset_infos.num_node_types, self.dataset_infos.num_edge_types)
        z_t, z_tilde, qt0, rate = self.noise_model.apply_noise(dense_data, validation=True)
        extra_data = self.compute_extra_data(z_tilde)

        pred = self.forward(z_tilde, extra_data)
        loss = self.val_loss(preds=pred, z_0=dense_data, z_tilde=z_tilde, qt0=qt0, rate=rate)

        return loss

    def test_step(self, data, i):
        dense_data = utils.to_dense(data, self.dataset_infos.num_node_types, self.dataset_infos.num_edge_types)
        z_t = self.noise_model.apply_noise(dense_data)

        extra_data = self.compute_extra_data(z_t)
        pred = self.forward(z_t, extra_data)

        dense_data = utils.to_dense(data, self.dataset_infos.num_node_types, self.dataset_infos.num_edge_types)
        z_t, z_tilde, qt0, rate = self.noise_model.apply_noise(dense_data, validation=True)
        extra_data = self.compute_extra_data(z_tilde)

        pred = self.forward(z_tilde, extra_data)
        loss = self.test_loss(preds=pred, z_0=dense_data, z_tilde=z_tilde, qt0=qt0, rate=rate)
        return loss


class DiffusionModelBucketDataloader(AbstractDiffusionModel):
    automatic_optimization = False
    ddp = None

    def __init__(self, cfg, dataset_infos, train_metrics, val_sampling_metrics, test_sampling_metrics,
                 visualization_tools, extra_features, domain_features):
        super().__init__(cfg, dataset_infos, train_metrics, val_sampling_metrics, test_sampling_metrics,
                 visualization_tools, extra_features, domain_features)

    def training_step(self, data, i):
        data = [d.to(self.device) if d is not None else None for d in data]
        optimizers = self.optimizers()
        optimizers.zero_grad()

        non_empties_buckets = [x for x in data if x is not None]
        bucket_sizes = [b.num_graphs for b in non_empties_buckets]
        log_dict = {}

        for i, bucket in enumerate(non_empties_buckets):
            with self.trainer.model.no_sync() if self.ddp else nullcontext():
                # Scale bucket loss proportionally to its size in the batch
                bucket_loss, bucket_log_dict = self.bucket_step(bucket)
                scale = bucket_sizes[i] / (self.BS if not self.trainer.is_last_batch else sum(bucket_sizes))
                self.manual_backward(bucket_loss * scale)

                # Scale bucket losses and add them to global losses
                bucket_log_dict = {key: value * scale for key, value in bucket_log_dict.items()}
                for key, value in bucket_log_dict.items():
                    if key in log_dict.keys():
                        log_dict[key] += value
                    else:
                        log_dict[key] = value

        optimizers.step()

        # Log losses and metrics manually
        if wandb.run and i % self.log_every_steps == 0:
            wandb.log(log_dict, commit=False)

    def bucket_step(self, bucket):
        if bucket.edge_index.numel() == 0:
            print("Found a batch with no edges. Skipping.")
            return
        dense_data = utils.to_dense(bucket, self.dataset_infos.num_node_types, self.dataset_infos.num_edge_types)

        z_t = self.noise_model.apply_noise(dense_data)
        extra_data = self.compute_extra_data(z_t)

        pred = self.forward(z_t, extra_data)
        loss, tl_log_dict = self.train_loss(preds=pred, z_0=dense_data, log=-1) # bucket logging disabled

        tm_log_dict = self.train_metrics(masked_pred=pred, masked_true=dense_data, log=-1)

        bucket_logdict = {}
        if tl_log_dict is not None:
            bucket_logdict.update(**tl_log_dict)
        if tm_log_dict is not None:
            bucket_logdict.update(**tm_log_dict)
        return loss, bucket_logdict

    def validation_step(self, data, i):
        data = [d.to(self.device) if d is not None else None for d in data]
        non_empties_buckets = [x for x in data if x is not None]

        for i, bucket in enumerate(non_empties_buckets):
            with self.trainer.model.no_sync() if self.ddp else nullcontext():
                # Scale bucket loss proportionally to its size in the batch
                _ = self.val_bucket_step(bucket)

    def test_step(self, data, i):
        data = [d.to(self.device) if d is not None else None for d in data]
        non_empties_buckets = [x for x in data if x is not None]

        for i, bucket in enumerate(non_empties_buckets):
            with self.trainer.model.no_sync() if self.ddp else nullcontext():
                # Scale bucket loss proportionally to its size in the batch
                _ = self.val_bucket_step(bucket, val=False)

    def val_bucket_step(self, data, val=True):
        dense_data = utils.to_dense(data, self.dataset_infos.num_node_types, self.dataset_infos.num_edge_types)
        z_t, z_tilde, qt0, rate = self.noise_model.apply_noise(dense_data, validation=True)
        extra_data = self.compute_extra_data(z_tilde)

        pred = self.forward(z_tilde, extra_data)
        loss = self.val_loss(preds=pred, z_0=dense_data, z_tilde=z_tilde, qt0=qt0, rate=rate) if val \
            else self.test_loss(preds=pred, z_0=dense_data, z_tilde=z_tilde, qt0=qt0, rate=rate)
        return loss
