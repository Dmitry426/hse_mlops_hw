import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig

from hse_mlops_hw import PROJECT_ROOT
from hse_mlops_hw.callbacks.predictors import CSVPredictionWriter
from hse_mlops_hw.data.data import MyDataModule
from hse_mlops_hw.models.model import MyModel
from hse_mlops_hw.services.getter import BestModelGetter


class Inference:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def infer(self):
        pl.seed_everything(42)

        output_folder = PROJECT_ROOT / "data"

        model = BestModelGetter(
            model=MyModel,
            experiment_path=output_folder / self.cfg.artifacts.experiment_name,
        ).get_best_model()

        data_module = MyDataModule(config=self.cfg)

        infer_model = pl.Trainer(
            accelerator=self.cfg.inference.accelerator,
            devices=self.cfg.inference.devices,
            precision=self.cfg.inference.precision,
            benchmark=self.cfg.inference.benchmark,
            inference_mode=self.cfg.inference.inference_mode,
            log_every_n_steps=self.cfg.inference.log_every_n_steps,
            default_root_dir=output_folder,
            callbacks=CSVPredictionWriter(
                output_dir=output_folder / "logs/infer_csv",
                write_interval="batch_and_epoch",
                name=self.cfg.artifacts.experiment_name,
            ),
        )

        infer_model.predict(model=model, datamodule=data_module)


@hydra.main(
    config_path=str(PROJECT_ROOT / "configs"),
    config_name="config.yaml",
    version_base="1.3",
)
def infer(cfg: DictConfig) -> None:
    Inference(cfg).infer()


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    infer()
