__all__ = "MyDataModule"

import logging
import os
import subprocess
import sys

import lightning.pytorch as pl
import omegaconf
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

logger = logging.getLogger(__name__)


class CatDogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ["cat", "dog"]
        self.data = []

        for class_label in self.classes:
            class_path = os.path.join(root_dir, class_label)
            class_idx = self.classes.index(class_label)
            for img_name in os.listdir(str(class_path)):
                img_path = os.path.join(str(class_path), img_name)
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
    def __init__(self, config: omegaconf.dictconfig.DictConfig):
        super().__init__()
        self.config = config
        self.train_folder = config.data.train_folder
        self.test_folder = config.data.test_folder
        self.val_size = config.data.val_size
        self.batch_size = config.data.batch_size
        self.dataloader_num_workers = config.data.dataloader_num_workers

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        self.classes = ["cat", "dog"]
        self.train_dataset = None
        self.val_dataset = None
        self.predict_dataset = None

    def prepare_data(self):
        try:
            subprocess.run(["dvc", "pull"], check=True)
            logger.info("DVC pull completed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error: DVC pull failed with exit code {e.returncode}.")
            sys.exit(e.returncode)
        except OSError as e:
            logger.error(f"Error: An unexpected error occurred - {e}")
            sys.exit(1)

    def setup(self, stage=None):
        train_dataset = CatDogDataset(
            root_dir=self.train_folder, transform=self.transform
        )
        train_size = int((1.0 - self.val_size) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            train_dataset, [train_size, val_size]
        )

        self.predict_dataset = CatDogDataset(
            root_dir=self.test_folder, transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.dataloader_num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dataloader_num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dataloader_num_workers,
        )
