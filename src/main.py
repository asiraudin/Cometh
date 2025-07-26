from rdkit import Chem
try:
    import graph_tool.all as gt
except ModuleNotFoundError:
    print("graph_tool not found, won't work")

import os
# os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
import torch

import pytorch_lightning as pl


import warnings
import pathlib

import hydra
from hydra.utils import to_absolute_path
import omegaconf

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from analysis.visualization import NonMolecularVisualization
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
from metrics.molecular_metrics import TrainMolecularMetrics
from metrics.sampling_metrics import SamplingMetrics
from analysis.visualization import MolecularVisualization
from diffusion.extra_features_molecular import ExtraMolecularFeatures

from diffusion_models import DiffusionModel, DiffusionModelBucketDataloader
from diffusion.extra_features import ExtraFeatures, DummyExtraFeatures
from ema import EMA, EMAModelCheckpoint

warnings.filterwarnings("ignore", category=PossibleUserWarning)
os.environ["WANDB__SERVICE_WAIT"] = "300"


def get_resume(cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools,
               extra_features, domain_features, checkpoint_path, test: bool):
    name = cfg.general.name + ('_test' if test else '_resume')
    gpus = cfg.general.gpus
    model = DiffusionModel.load_from_checkpoint(checkpoint_path, dataset_infos=dataset_infos,
                                                train_metrics=train_metrics, sampling_metrics=sampling_metrics,
                                                visualization_tools=visualization_tools, extra_features=extra_features,
                                                domain_features=domain_features)
    cfg.general.gpus = gpus
    cfg.general.name = name
    return cfg, model


@hydra.main(version_base='1.3.2', config_path='../configs', config_name='config')
def main(cfg: omegaconf.DictConfig):
    dataset_config = cfg.dataset
    pl.seed_everything(cfg.train.seed)

    if dataset_config.name in ['qm9', 'guacamol', 'moses']:
        from datasets import qm9_dataset, moses_dataset, guacamol_dataset
        if dataset_config.name == 'qm9':
            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
        elif dataset_config.name == 'guacamol':
            datamodule = guacamol_dataset.GuacamolDataModule(cfg)
            dataset_infos = guacamol_dataset.Guacamolinfos(datamodule=datamodule, cfg=cfg)
        elif dataset_config.name == 'moses':
            datamodule = moses_dataset.MOSESDataModule(cfg=cfg)
            dataset_infos = moses_dataset.MOSESinfos(datamodule=datamodule, cfg=cfg)
        else:
            raise NotImplementedError("Dataset not implemented yet")
        dataloaders = None

        train_metrics = TrainMolecularMetrics(dataset_infos)
        domain_features = ExtraMolecularFeatures(cfg.encoding.molecular_features, dataset_infos=dataset_infos)
        visualization_tools = MolecularVisualization(remove_h=cfg.dataset.remove_h, dataset_infos=dataset_infos)

    elif dataset_config.name in ['planar', 'sbm', 'comm-20']:
        from datasets.spectre_dataset import SBMDataModule, Comm20DataModule, EgoDataModule, PlanarDataModule, \
            SpectreDatasetInfos

        if dataset_config["name"] == "sbm":
            datamodule = SBMDataModule(cfg)
        elif dataset_config["name"] == "comm20":
            datamodule = Comm20DataModule(cfg)
        elif dataset_config["name"] == "ego":
            datamodule = EgoDataModule(cfg)
        else:
            datamodule = PlanarDataModule(cfg)

        dataset_infos = SpectreDatasetInfos(datamodule)
        train_metrics = TrainAbstractMetricsDiscrete()
        visualization_tools = NonMolecularVisualization()
        domain_features = DummyExtraFeatures()
        dataloaders = datamodule.dataloaders

    extra_features = ExtraFeatures(cfg.encoding, dataset_info=dataset_infos)
    val_sampling_metrics = SamplingMetrics(
        dataset_infos, test=False, dataloaders=dataloaders
    )
    test_sampling_metrics = SamplingMetrics(
        dataset_infos, test=True, dataloaders=dataloaders
    )

    if dataset_config.bucketloader:
        model = DiffusionModelBucketDataloader(cfg=cfg, dataset_infos=dataset_infos, train_metrics=train_metrics,
                                               val_sampling_metrics=val_sampling_metrics,
                                               test_sampling_metrics=test_sampling_metrics,
                                               visualization_tools=visualization_tools, extra_features=extra_features,
                                               domain_features=domain_features)
    else:
        model = DiffusionModel(cfg=cfg, dataset_infos=dataset_infos, train_metrics=train_metrics,
                               val_sampling_metrics=val_sampling_metrics,
                               test_sampling_metrics=test_sampling_metrics,
                               visualization_tools=visualization_tools, extra_features=extra_features,
                               domain_features=domain_features)

    callbacks = []

    if cfg.train.save_model:
        checkpoint_callback = EMAModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                                 filename='{epoch}',
                                                 monitor='val/elbo',
                                                 save_top_k=10,
                                                 mode='min',)
        # fix a name and keep overwriting
        last_ckpt_save = EMAModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=1)
        callbacks.append(checkpoint_callback)
        callbacks.append(last_ckpt_save)
    if cfg.train.ema_decay > 0:
        ema_callback = EMA(decay=cfg.train.ema_decay,
                           save_ema_weights_in_callback_state=True,
                           evaluate_ema_weights_instead=True)
        callbacks.append(ema_callback)

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    name = cfg.general.name
    if name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    trainer = Trainer(accelerator='gpu' if use_gpu else 'cpu',
                      strategy='ddp' if name != 'debug' else 'auto',
                      devices=int(cfg.general.gpus) if use_gpu else 1,
                      callbacks=callbacks,
                      fast_dev_run=cfg.general.name == 'debug',
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      log_every_n_steps=50 if name != 'debug' else 1,
                      num_nodes=1,
                      enable_progress_bar=False,
                      gradient_clip_val=cfg.train.clip_grad,
                      logger=False)

    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=to_absolute_path(cfg.general.resume) \
            if cfg.general.resume is not None else cfg.general.resume)
    else:
        # Start by evaluating test_only_path
        for i in range(cfg.general.num_final_sampling):
            pl.seed_everything(cfg.general.final_seeds[i])
            trainer.test(model, datamodule=datamodule, ckpt_path=to_absolute_path(cfg.general.test_only))
        if cfg.general.evaluate_all_checkpoints:
            pl.seed_everything(cfg.train.seed)
            directory = pathlib.Path(to_absolute_path(cfg.general.test_only)).parents[0]
            print("Directory:", directory)
            files_list = os.listdir(directory)
            for file in sorted(files_list):
                if '.ckpt' in file and '-EMA' not in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == to_absolute_path(cfg.general.test_only):
                        continue
                    print("Loading checkpoint", ckpt_path)
                    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
