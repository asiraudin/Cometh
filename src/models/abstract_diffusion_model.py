import time
import os
import math

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb

import utils
from models.transformer_model import GraphTransformer
from diffusion.noise_model import UniformRateConstant, UniformRateCosine, MarginalRateConstant, MarginalRateCosine
from metrics.train_metrics import TrainLoss, ValidationLoss

from metrics.molecular_metrics import Molecule


class AbstractDiffusionModel(pl.LightningModule):
    model_dtype = torch.float32
    best_val_loss = 1e8
    val_counter = 0
    train_iterations = None
    start_epoch_time = None

    def __init__(self, cfg, dataset_infos, train_metrics, val_sampling_metrics, test_sampling_metrics,
                 visualization_tools, extra_features, domain_features):
        super().__init__()
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.T = cfg.model.diffusion_steps
        self.molecular_dataset = dataset_infos.name in ['qm9', 'moses', 'guacamol']

        self.node_dist = nodes_dist
        self.dataset_infos = dataset_infos

        self.extra_features = extra_features
        self.input_dims = self.extra_features.update_input_dims(dataset_infos.input_dims)

        self.output_dims = dataset_infos.output_dims
        self.domain_features = domain_features
        self.input_dims = self.domain_features.update_input_dims(self.input_dims)

        self.train_loss = TrainLoss(train=True,
                                    lambda_train=self.cfg.model.lambda_train
                                    if hasattr(self.cfg.model, "lambda_train") else self.cfg.train.lambda0)
        self.train_metrics = train_metrics

        self.val_sampling_metrics = val_sampling_metrics
        self.test_sampling_metrics = test_sampling_metrics
        self.val_loss = ValidationLoss()
        self.test_loss = ValidationLoss()

        self.save_hyperparameters(ignore=['train_metrics', 'val_sampling_metrics', 'test_sampling_metrics',
                                          'dataset_infos'])

        self.visualization_tools = visualization_tools
        self.model = GraphTransformer(input_dims=self.input_dims,
                                      n_layers=cfg.model.n_layers,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=self.output_dims,
                                      encoding_config=cfg.encoding)

        if cfg.model.transition == "uniform":
            if cfg.model.schedule == "cosine":
                self.noise_model = UniformRateCosine(cfg=cfg, output_dims=self.output_dims)
            elif cfg.model.schedule == "constant":
                self.noise_model = UniformRateConstant(cfg=cfg, output_dims=self.output_dims)
        elif cfg.model.transition == "marginal":
            print(f"Marginal distribution of the classes: nodes: {self.dataset_infos.node_types} --"
                  f" edges: {self.dataset_infos.edge_types}")
            if cfg.model.schedule == "cosine":
                self.noise_model = MarginalRateCosine(cfg=cfg,
                                                      x_marginals=self.dataset_infos.node_types,
                                                      e_marginals=self.dataset_infos.edge_types,
                                                      y_classes=self.output_dims.y)
            elif cfg.model.schedule == "constant":
                self.noise_model = MarginalRateConstant(cfg=cfg,
                                                        x_marginals=self.dataset_infos.node_types,
                                                        e_marginals=self.dataset_infos.edge_types,
                                                        y_classes=self.output_dims.y)
                assert False
        else:
            raise NotImplementedError("Noise model other than uniform not implemented yet.")

        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps

        self.min_time = cfg.model.min_time
        self.corrector_entry_time = cfg.model.corrector_entry_time
        self.corrector_num_steps = cfg.model.corrector_num_steps

    def on_train_epoch_start(self) -> None:
        self.print("Starting epoch", self.current_epoch)
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        self.print(f"Train epoch {self.current_epoch} ends")
        tle_log = self.train_loss.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch} finished: "
                   f"X: {tle_log['train_epoch/X_loss'] :.2f} --"
                   f" E: {tle_log['train_epoch/E_loss'] :.2f} --"
                   f" {time.time() - self.start_epoch_time:.1f}s ")

        if wandb.run:
            wandb.log({"epoch": self.current_epoch}, commit=False)

    def on_validation_epoch_start(self) -> None:
        self.val_loss.reset()
        self.val_sampling_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        to_log = self.val_loss.log_epoch_metrics()

        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        val_elbo = to_log["val_epoch/elbo"]
        self.log("val/elbo", val_elbo, sync_dist=True)

        if val_elbo < self.best_val_loss:
            self.best_val_loss = val_elbo
        print('Val loss: %.4f \t Best val loss:  %.4f\n' % (val_elbo, self.best_val_loss))

        self.val_counter += 1
        if self.name == "debug" or (self.val_counter % self.cfg.general.sample_every_val == 0 and
                                    self.current_epoch > 0):
            self.print(f"Sampling start")
            start = time.time()
            gen = self.cfg.general
            samples = self.sample_n_graphs(samples_to_generate=math.ceil(gen.samples_to_generate / max(gen.gpus, 1)),
                                           chains_to_save=gen.chains_to_save if self.local_rank == 0 else 0,
                                           samples_to_save=gen.samples_to_save if self.local_rank == 0 else 0)
            print(f'Done on {self.local_rank}. Sampling took {time.time() - start:.2f} seconds\n')
            print(f"Computing sampling metrics on {self.local_rank}...")
            self.val_sampling_metrics.compute_all_metrics(samples,
                                                          current_epoch=self.current_epoch,
                                                          local_rank=self.local_rank)
        self.print(f"Val epoch {self.current_epoch} ends")

    def on_test_epoch_start(self):
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)
        self.test_loss.reset()
        self.test_sampling_metrics.reset()

    def on_test_epoch_end(self) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        self.test_loss.log_epoch_metrics()

        print(f"Sampling start on GR{self.global_rank}")
        start = time.time()
        print(f"Samples to generate: {self.cfg.general.final_model_samples_to_generate}")
        print(f"Samples to save: {self.cfg.general.final_model_samples_to_save}")
        samples = self.sample_n_graphs(samples_to_generate=self.cfg.general.final_model_samples_to_generate,
                                       chains_to_save=self.cfg.general.final_model_chains_to_save,
                                       samples_to_save=self.cfg.general.final_model_samples_to_save)
        print("Saving the generated graphs")
        filename = f'generated_samples1.txt'
        for i in range(2, 10):
            if os.path.exists(filename):
                filename = f'generated_samples{i}.txt'
            else:
                break
        with open(filename, 'w') as f:
            for graph in samples:
                f.write(f"N={len(graph[0])}\n")
                # X:
                nodes = graph[0]
                f.write("X: \n")
                for node in nodes:
                    f.write(f"{node} ")
                f.write("\n")

                f.write("E: \n")
                for edge_list in graph[1]:
                    for edge in edge_list:
                        f.write(f"{edge} ")
                    f.write("\n")
                f.write("\n")
        print("Saved.")
        print("Computing sampling metrics...")
        self.test_sampling_metrics.compute_all_metrics(samples, self.current_epoch, self.local_rank)
        print(f'Done. Sampling took {time.time() - start:.2f} seconds\n')
        print(f"Test ends.")

    def sample_n_graphs(self, samples_to_generate: int, chains_to_save: int, samples_to_save: int):
        samples_left_to_generate = samples_to_generate
        samples_left_to_save = samples_to_save
        chains_left_to_save = chains_to_save

        samples = []

        ident = 0
        while samples_left_to_generate > 0:
            bs = 2 * self.cfg.train.batch_size
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            samples.extend(self.sample_batch(batch_id=ident, batch_size=to_generate, num_nodes=None,
                                             save_final=to_save,
                                             keep_chain=chains_save,
                                             number_chain_steps=self.number_chain_steps))
            ident += to_generate

            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save

        return samples

    @torch.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
                     save_final: int, num_nodes=None):
        """
        :param batch_id: int
        :param n_nodes: list of int containing the number of nodes to sample for each graph
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: graph_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        # print(f"Sampling a batch with {len(n_nodes)} graphs. Saving {save_final} visualization and {keep_chain} full chains.")
        assert keep_chain >= 0
        assert save_final >= 0
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes

        n_nodes = torch.Tensor(n_nodes).long().to(self.device)

        batch_size = len(n_nodes)
        n_max = torch.max(n_nodes).detach()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)

        # Sample noise -- z has size (n_samples, n_nodes, n_features)
        z_T = self.noise_model.sample_limit_dist(node_mask=node_mask)

        assert (z_T.E == torch.transpose(z_T.E, 1, 2)).all()
        assert number_chain_steps < self.T

        n_max = z_T.X.size(1)
        chains = utils.PlaceHolder(X=torch.zeros((number_chain_steps, keep_chain, n_max), dtype=torch.long),
                                   E=torch.zeros((number_chain_steps, keep_chain, n_max, n_max)),
                                   y=None)

        z_t = z_T

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for t_int in reversed(range(1, self.T + 1)):
            z_s = self.sample_zs_from_zt(z_t=z_t)

            if z_t.t[0] <= (1.0 - self.min_time) * self.corrector_entry_time + self.min_time:
                for _ in range(self.corrector_num_steps):
                    z_s = self.sample_zs_from_zt(z_t=z_s, corrector=True)

            # Save the first keep_chain graphs
            if ((t_int-1) * number_chain_steps) % self.T == 0:
                write_index = number_chain_steps - 1 - ((t_int * number_chain_steps) // self.T)
                discrete_z_s = z_s.collapse()
                chains.X[write_index] = discrete_z_s.X[:keep_chain]
                chains.E[write_index] = discrete_z_s.E[:keep_chain]

            z_t = z_s

        # Last network pass at t_min
        z_0 = self.sample_zs_from_zt(z_t, last_pass=True)

        # Sample final data
        sampled = z_0.collapse()
        X, E, y = sampled.X, sampled.E, sampled.y

        chains.X[-1] = X[:keep_chain]  # Overwrite last frame with the resulting X, E
        chains.E[-1] = E[:keep_chain]

        graph_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n]
            edge_types = E[i, :n, :n]
            graph_list.append([atom_types, edge_types])

            # Visualize chains
            # if self.visualization_tools is not None:
            #     # self.print('Visualizing chains...')
            #     current_path = os.getcwd()
            #     num_molecules = chains.X.size(1)  # number of molecules
            #     for i in range(num_molecules):
            #         result_path = os.path.join(current_path, f'chains/{self.cfg.general.name}/'
            #                                                  f'epoch{self.current_epoch}/'
            #                                                  f'chains/molecule_{batch_id + i}')
                    # if not os.path.exists(result_path):
                    #     os.makedirs(result_path)
                    #     _ = self.visualization_tools.visualize_chain(result_path,
                    #                                                  chains.X[:, i, :].numpy(),
                    #                                                   chains.E[:, i, :].numpy())
                    # self.print('\r{}/{} complete'.format(i + 1, num_molecules), end='', flush=True)
                # self.print('\nVisualizing molecules...')

                # # Visualize the final molecules
                # current_path = os.getcwd()
                # result_path = os.path.join(current_path,
                #                            f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
                # self.visualization_tools.visualize(result_path, graph_list, save_final)
                # # self.print("Done.")
        return graph_list

    def sample_zs_from_zt(self, z_t, corrector=False, last_pass=False):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        extra_data = self.compute_extra_data(z_t)
        preds = self.forward(z_t, extra_data)
        z_s = self.noise_model.sample_zs_from_zt_and_pred(z_t=z_t, preds=preds,
                                                          last_pass=last_pass, corrector=corrector)
        return z_s

    @property
    def BS(self):
        return self.cfg.train.batch_size

    def forward(self, z_tilde, extra_data):
        assert z_tilde.node_mask is not None
        model_input = z_tilde.copy()
        model_input.X = torch.cat((z_tilde.X, extra_data.X), dim=2).float()
        model_input.E = torch.cat((z_tilde.E, extra_data.E), dim=3).float()
        model_input.y = torch.hstack((z_tilde.y, z_tilde.t, extra_data.y)).float()
        return self.model(model_input)

    def compute_extra_data(self, z_t):
        extra_features = self.extra_features(z_t)
        extra_molecular_features = self.domain_features(z_t)

        extra_X = torch.cat((extra_molecular_features.X, extra_features.X), dim=-1)
        extra_E = torch.cat((extra_molecular_features.E, extra_features.E), dim=-1)
        extra_y = torch.cat((extra_molecular_features.y, extra_features.y), dim=-1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)
