__all__ = "CSVPredictionWriter"

import csv
import logging
import os
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import lightning.pytorch as pl
import torch
from lightning.fabric.loggers import CSVLogger
from lightning.pytorch.callbacks import BasePredictionWriter

logger = logging.getLogger(__name__)


# pylint: disable=unused-argument
class CSVPredictionWriter(BasePredictionWriter, CSVLogger):
    """Csv prediction writer"""

    def __init__(
        self,
        output_dir: str,
        name: str,
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
        torch.save(
            prediction,
            self.get_writing_path(f"batch_{batch_idx}.pt"),
        )
        self.log_predictions_to_csv(prediction, batch_idx, name="predictions.csv")

    def write_on_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        predictions: Sequence[Any],
        batch_indices: Sequence[Any],
    ) -> None:
        torch.save(
            predictions,
            self.get_writing_path("predictions.pt"),
        )

        if self.interval == "epoch":
            self.log_predictions_to_csv(predictions, name="predictions.csv")

    def get_writing_path(self, filename: Optional[str]) -> str:
        """Get writing path and insure its existence before writing"""
        path = self.output_dir / self.experiment_name / self.get_version
        self.ensure_path(path)
        return str(path / filename)

    def log_predictions_to_csv(
        self,
        predictions: Sequence[Any],
        batch_idx: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        path = self.get_writing_path(name)
        with open(path, "a", newline="", encoding="utf8") as csv_file:
            csv_writer = csv.writer(csv_file)
            for pred in predictions:
                row = {"Prediction": pred, "BatchIndex": batch_idx}
                csv_writer.writerow(row.values())

    @staticmethod
    def ensure_path(directory_path: Path):
        """Check if directory and do creation if it doesn't exist"""
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            logger.info(f"Directory '{directory_path}' created.")
