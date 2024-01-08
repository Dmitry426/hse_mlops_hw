import logging
import os
import re
from pathlib import Path
from typing import List

import lightning.pytorch as pl

logger = logging.getLogger(__name__)


class BestModelGetter:
    """Get best checkpoint by loss for inference"""

    def __init__(self, model, experiment_path: str | Path, loss: str = "val_loss"):
        self.model = model
        self.experiment_path = Path(experiment_path)
        self.loss = loss

    def get_best_model(self) -> pl.LightningModule:
        """Get the best model checkpoint for inference"""
        best_checkpoint = self.get_best_checkpoint_on_l_loss()

        checkpoint_path = self.experiment_path / best_checkpoint

        model = self.model.load_from_checkpoint(str(checkpoint_path))

        return model

    def get_best_checkpoint_on_l_loss(self) -> str:
        """
        Get the file name with the lowest val_loss value from a list of file names.
        """
        saved_checkpoints = self._list_files_in_experiment(self.experiment_path)
        val_loss_values = self._get_loss_values(saved_checkpoints, loss=self.loss)

        min_val_loss_index = val_loss_values.index(min(val_loss_values))
        best_file_name = saved_checkpoints[min_val_loss_index]

        logging.info(f"Loaded best model checkpoint {best_file_name}")

        return best_file_name

    @staticmethod
    def _list_files_in_experiment(folder_path: Path) -> List[str]:
        """
        List files in the specified folder.
        """
        try:
            files = os.listdir(folder_path)
            return files
        except FileNotFoundError as err:
            logger.error(
                f"The specified folder '{folder_path}' does not include experiments."
            )
            raise err

    @staticmethod
    def _get_loss_values(file_names: List[str], loss) -> List[float]:
        """
        Get a list of val_loss values from a list of file names.
        """
        val_loss_values = []

        for file_name in file_names:
            match = re.search(rf"{loss}=(\d+\.\d+)", file_name)
            if match:
                val_loss_values.append(float(match.group(1)))

        return val_loss_values
