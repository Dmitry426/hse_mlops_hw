import os

import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig

from hse_mlops_hw import PROJECT_ROOT
from hse_mlops_hw.callbacks.predictors import CSVPredictionWriter
from hse_mlops_hw.data.data import MyDataModule
from hse_mlops_hw.models.model import MyModel


@hydra.main(config_path="../configs", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    pl.seed_everything(42)

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
        num_sanity_val_steps=cfg.train.num_sanity_val_steps,
        benchmark=cfg.train.benchmark,
        inference_mode=True,
        log_every_n_steps=10,
        enable_checkpointing=False,
        default_root_dir=output_folder,
        callbacks=CSVPredictionWriter(
            output_dir=f"{output_folder}/logs/infer_csv",
            write_interval="batch_and_epoch",
            name=cfg.artifacts.experiment_name,
        ),
    )

    infer.predict(model=model, datamodule=data_module)


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    main()
