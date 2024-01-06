from pathlib import Path
from typing import Union

import lightning.pytorch as pl
from omegaconf import DictConfig, ListConfig, OmegaConf

from hse_mlops_hw import PROJECT_ROOT
from hse_mlops_hw.callbacks.predictors import CSVPredictionWriter
from hse_mlops_hw.data.data import MyDataModule
from hse_mlops_hw.models.model import MyModel
from hse_mlops_hw.services.getter import BestModelGetter


class Inference:
    def __init__(self, cfg: Union[DictConfig, ListConfig]):
        self.cfg = cfg

    def infer(self):
        pl.seed_everything(42)

        output_folder = Path(PROJECT_ROOT) / "data"

        model = BestModelGetter(
            model=MyModel,
            experiment_path=output_folder / self.cfg.artifacts.experiment_name,
        ).get_best_model()

        data_module = MyDataModule(config=self.cfg)

        infer_model = pl.Trainer(
            max_epochs=1,
            accelerator=self.cfg.train.accelerator,
            devices=self.cfg.train.devices,
            precision=self.cfg.train.precision,
            benchmark=self.cfg.train.benchmark,
            inference_mode=True,
            log_every_n_steps=self.cfg.train.log_every_n_steps,
            default_root_dir=output_folder,
            callbacks=CSVPredictionWriter(
                output_dir=output_folder / "logs/infer_csv",
                write_interval="batch_and_epoch",
                name=self.cfg.artifacts.experiment_name,
            ),
        )

        infer_model.predict(model=model, datamodule=data_module)


def infer(
    config_path: str = "../configs/config.yaml", version_base: str = "1.3"
) -> None:
    cfg = OmegaConf.load(config_path)
    cfg.version_base = version_base

    Inference(cfg).infer()


if __name__ == "__main__":
    infer()
