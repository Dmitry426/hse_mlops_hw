import os

import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig

from hse_mlops_hw import PROJECT_ROOT
from hse_mlops_hw.data.data import MyDataModule
from hse_mlops_hw.models.model import MyModel


@hydra.main(config_path="../configs", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    output_folder = os.path.join(PROJECT_ROOT, "data")
    experiment_folder = os.path.join(output_folder, cfg.artifacts.experiment_name)

    # pylint: disable=no-value-for-parameter
    model = MyModel.load_from_checkpoint(
        checkpoint_path=os.path.join(experiment_folder, "epoch=00-val_loss=0.9038.ckpt")
    )

    data_module = MyDataModule(config=cfg)

    infer = pl.Trainer(
        max_epochs=1,
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
        default_root_dir=output_folder,
    )

    infer.predict(model=model, datamodule=data_module)


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    main()
