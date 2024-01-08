from pathlib import Path

import hydra
import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import CSVLogger, MLFlowLogger
from lightning.pytorch.tuner import Tuner
from omegaconf import DictConfig

from hse_mlops_hw import PROJECT_ROOT
from hse_mlops_hw.data.data import MyDataModule
from hse_mlops_hw.models.model import MyModel


class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def train(self) -> None:
        pl.seed_everything(42)

        torch.set_float32_matmul_precision("medium")
        dm = MyDataModule(self.cfg)

        model = MyModel(self.cfg)

        output_folder = Path(PROJECT_ROOT) / "data"

        loggers = [
            CSVLogger(
                f"{output_folder}/logs/train_csv",
                name=self.cfg.artifacts.experiment_name,
            ),
            MLFlowLogger(
                experiment_name=self.cfg.artifacts.experiment_name,
                tracking_uri=self.cfg.artifacts.tracking_uri,
            ),
            pl.loggers.TensorBoardLogger(
                f"{output_folder}/logs/tensorboard",
                name=self.cfg.artifacts.experiment_name,
            ),
        ]

        callbacks = [
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            pl.callbacks.DeviceStatsMonitor(),
            pl.callbacks.RichModelSummary(
                max_depth=self.cfg.callbacks.model_summary.max_depth
            ),
        ]

        if self.cfg.artifacts.checkpoint.use:
            callbacks.append(
                pl.callbacks.ModelCheckpoint(
                    dirpath=output_folder
                    / self.cfg.artifacts.checkpoint.dirpath
                    / self.cfg.artifacts.experiment_name,
                    filename=self.cfg.artifacts.checkpoint.filename,
                    monitor=self.cfg.artifacts.checkpoint.monitor,
                    save_top_k=self.cfg.artifacts.checkpoint.save_top_k,
                    every_n_train_steps=self.cfg.artifacts.checkpoint.every_n_train,
                    every_n_epochs=self.cfg.artifacts.checkpoint.every_n_epochs,
                )
            )

        trainer = pl.Trainer(
            max_epochs=self.cfg.train.max_epochs,
            limit_train_batches=self.cfg.train.limit_train_batches,
            accelerator=self.cfg.train.accelerator,
            devices=self.cfg.train.devices,
            precision=self.cfg.train.precision,
            max_steps=self.cfg.train.num_warmup_steps
            + self.cfg.train.num_training_steps,
            accumulate_grad_batches=self.cfg.train.grad_accum_steps,
            val_check_interval=self.cfg.train.val_check_interval,
            overfit_batches=self.cfg.train.overfit_batches,
            num_sanity_val_steps=self.cfg.train.num_sanity_val_steps,
            deterministic=self.cfg.train.full_deterministic_mode,
            benchmark=self.cfg.train.benchmark,
            gradient_clip_val=self.cfg.train.gradient_clip_val,
            profiler=self.cfg.train.profiler,
            log_every_n_steps=self.cfg.train.log_every_n_steps,
            detect_anomaly=self.cfg.train.detect_anomaly,
            enable_checkpointing=self.cfg.artifacts.checkpoint.use,
            logger=loggers,
            callbacks=callbacks,
            default_root_dir=output_folder,
        )

        if self.cfg.train.batch_size_finder:
            tuner = Tuner(trainer)
            tuner.scale_batch_size(model, datamodule=dm, mode="power")

        trainer.fit(model, datamodule=dm)


@hydra.main(
    config_path=str(PROJECT_ROOT / "configs"),
    config_name="config.yaml",
    version_base="1.3",
)
def train(cfg: DictConfig) -> None:
    Trainer(cfg).train()


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    train()
