import os

import hydra
import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import CSVLogger, MLFlowLogger
from lightning.pytorch.tuner import Tuner
from omegaconf import DictConfig

from hse_mlops_hw import PROJECT_ROOT
from hse_mlops_hw.data.data import MyDataModule
from hse_mlops_hw.models.model import MyModel


@hydra.main(config_path="../configs", config_name="config.yaml", version_base="1.3")
def train(cfg: DictConfig):
    pl.seed_everything(42)
    torch.set_float32_matmul_precision("medium")
    dm = MyDataModule(cfg)

    model = MyModel(cfg)

    output_folder = os.path.join(PROJECT_ROOT, "data")

    loggers = [
        CSVLogger(
            f"{output_folder}/logs/my-csv-logs", name=cfg.artifacts.experiment_name
        ),
        MLFlowLogger(
            experiment_name=cfg.artifacts.experiment_name,
            tracking_uri=cfg.artifacts.tracking_uri,
        ),
    ]

    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.RichModelSummary(max_depth=cfg.callbacks.model_summary.max_depth),
    ]

    if cfg.artifacts.checkpoint.use:
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=str(
                    os.path.join(
                        cfg.artifacts.checkpoint.dirpath, cfg.artifacts.experiment_name
                    )
                ),
                filename=cfg.artifacts.checkpoint.filename,
                monitor=cfg.artifacts.checkpoint.monitor,
                save_top_k=cfg.artifacts.checkpoint.save_top_k,
                every_n_train_steps=cfg.artifacts.checkpoint.every_n_train_steps,
                every_n_epochs=cfg.artifacts.checkpoint.every_n_epochs,
            )
        )

    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=0.2,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        max_steps=cfg.train.num_warmup_steps + cfg.train.num_training_steps,
        accumulate_grad_batches=cfg.train.grad_accum_steps,
        val_check_interval=cfg.train.val_check_interval,
        overfit_batches=cfg.train.overfit_batches,
        num_sanity_val_steps=cfg.train.num_sanity_val_steps,
        deterministic=cfg.train.full_deterministic_mode,
        benchmark=cfg.train.benchmark,
        gradient_clip_val=cfg.train.gradient_clip_val,
        profiler=cfg.train.profiler,
        log_every_n_steps=cfg.train.log_every_n_steps,
        detect_anomaly=cfg.train.detect_anomaly,
        enable_checkpointing=cfg.artifacts.checkpoint.use,
        logger=loggers,
        callbacks=callbacks,
        default_root_dir=output_folder,
    )

    if cfg.train.batch_size_finder:
        tuner = Tuner(trainer)
        tuner.scale_batch_size(model, datamodule=dm, mode="power")

    trainer.fit(model, datamodule=dm)


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    train()
