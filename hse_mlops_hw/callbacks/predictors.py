__all__ = "CSVPredictionWriter"

import logging
import os
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import lightning.pytorch as pl
import pandas as pd
import torch
from lightning.fabric.loggers import CSVLogger
from lightning.pytorch.callbacks import BasePredictionWriter

logger = logging.getLogger(__name__)


# pylint: disable=unused-argument
class CSVPredictionWriter(BasePredictionWriter, CSVLogger):
    """
    Csv prediction writer that extends BasePredictionWriter and CSVLogger.

    Args:
        output_dir (Union[Path, str]): The directory where the CSV predictions will be
        saved.

        name (str, optional): The name of the CSV file (without extension). Defaults to
         "predictions".

        write_interval (Literal["batch", "epoch", "batch_and_epoch"], optional):
            Determines when to write predictions to CSV during training.
            Options are "batch", "epoch", or "batch_and_epoch". Defaults to "batch".

    Properties:
        get_version (property): Returns the version if it's a string, otherwise,
        returns "version_{self.version}".

    Example:
        ```python
        from your_module import CSVPredictionWriter

        csv_writer = CSVPredictionWriter(output_dir="/path/to/csv",
         name="my_predictions")
        ```
    """

    def __init__(
        self,
        output_dir: Path | str,
        name: str = "predictions",
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
    ):
        super().__init__(write_interval=write_interval)
        CSVLogger.__init__(self, root_dir=output_dir, name=name)
        self.output_dir = Path(output_dir)
        self.experiment_name = name

    @property
    def get_version(self):
        return (
            self.version if isinstance(self.version, str) else f"version_{self.version}"
        )

    def write_on_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Write predictions to csv on batch ends"""
        self.log_predictions_to_csv(prediction, name=f"{self.name}.csv")

    def write_on_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        predictions: dict,  # type: ignore[override]
        batch_indices: Sequence[Any],
    ) -> None:
        """Write predictions to csv on epoch ends"""
        if self.interval == "epoch":
            self.log_predictions_to_csv(predictions, name=f"{self.name}.csv")

    def get_writing_path(self, filename: Optional[str]) -> str:
        """Get writing path and ensure its existence before writing"""
        path = self.output_dir / self.experiment_name / self.get_version
        self.ensure_path(path)
        return str(path / filename)

    def log_predictions_to_csv(
        self,
        predictions: dict,
        name: Optional[str] = None,
    ) -> None:
        """
        Log predictions to a CSV file.
        """
        path = self.get_writing_path(name)

        if predictions["logits"].dtype == torch.bfloat16:
            logits = (
                predictions["logits"]
                .to(dtype=torch.float16)
                .cpu()
                .numpy(force=True)
                .tolist()
            )
        else:
            logits = predictions["logits"].cpu().numpy(force=True).tolist()

        if predictions["probabilities"].dtype == torch.bfloat16:
            probabilities = (
                predictions["probabilities"]
                .to(dtype=torch.float16)
                .cpu()
                .numpy(force=True)
                .tolist()
            )
        else:
            probabilities = (
                predictions["probabilities"].cpu().numpy(force=True).tolist()
            )

        df = pd.DataFrame({"logits": logits, "probabilities": probabilities})

        df.to_csv(path, mode="a", header=False, index=False)

    @staticmethod
    def ensure_path(directory_path: Path) -> None:
        """Check if directory and do creation if it doesn't exist"""
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            logger.info(f"Directory '{directory_path}' created.")
