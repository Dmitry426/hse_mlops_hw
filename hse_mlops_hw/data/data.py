__all__ = "MyDataModule"

import logging
import subprocess
import sys
from pathlib import Path
from typing import List

import lightning.pytorch as pl
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.transforms import Compose

from hse_mlops_hw import PROJECT_ROOT

logger = logging.getLogger(__name__)


class CatDogDataset(Dataset):
    def __init__(self, root_dir: Path, classes: List, transform: Compose = None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = classes
        self.data = []

        for class_label in self.classes:
            class_path = Path(root_dir) / class_label
            class_idx = self.classes.index(class_label)

            for img_name in class_path.iterdir():
                img_path = str(img_name)
                self.data.append((img_path, class_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_idx = self.data[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, class_idx


class MyDataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.train_path = self._ensure_path(config.data.train_path)
        self.test_path = self._ensure_path(config.data.test_path)
        self.val_size = config.data.val_size
        self.batch_size = config.data.batch_size
        self.dataloader_num_workers = config.data.dataloader_num_workers

        self.transform = Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        self.classes = config.data.labels
        self.train_dataset: CatDogDataset
        self.val_dataset: CatDogDataset
        self.predict_dataset: CatDogDataset

    def prepare_data(self):
        try:
            subprocess.run(["dvc", "pull"], check=True)
            logger.info("DVC pull completed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error: DVC pull failed with exit code {e.returncode}.")
        except OSError as e:
            logger.error(f"Error: An unexpected error occurred - {e}")
            sys.exit(1)

    def setup(self, stage=None):
        train_dataset = CatDogDataset(
            root_dir=self.train_path, transform=self.transform, classes=self.classes
        )
        train_size = int((1.0 - self.val_size) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            train_dataset, [train_size, val_size]
        )

        self.predict_dataset = CatDogDataset(
            root_dir=self.test_path, transform=self.transform, classes=self.classes
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.dataloader_num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dataloader_num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dataloader_num_workers,
        )

    @staticmethod
    def _ensure_path(path: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
        """
        Ensure that the given path exists and is
        absolute or relative to the project root.
        """
        path = Path(path)

        if path.is_absolute() and path.exists():
            return path
        else:
            absolute_path = project_root / path
            if absolute_path.exists():
                return absolute_path
            else:
                error_message = f"The path '{absolute_path}' does not exist."
                logging.error(error_message)
                raise FileNotFoundError(error_message)
