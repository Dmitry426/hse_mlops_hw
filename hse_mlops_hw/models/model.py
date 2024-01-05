__all__ = "MyModel"

from typing import Any

import lightning.pytorch as pl
import omegaconf
import torch
import transformers


# pylint: disable=unused-argument
class MyModel(pl.LightningModule):
    def __init__(self, conf: omegaconf.dictconfig.DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.conf = conf
        self.backbone = transformers.AutoModel.from_pretrained(
            conf.model.name,
            return_dict=True,
            output_hidden_states=True,
        )
        self.drop = torch.nn.Dropout(p=conf["model"]["dropout"])
        self.fc = torch.nn.LazyLinear(out_features=2)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, image, labels=None):
        embeddings = self.backbone(image)["last_hidden_state"][:, 0]
        embeddings = self.drop(embeddings)
        logits = self.fc(embeddings)

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return logits, loss
        return logits

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        image, labels = batch
        _, loss = self(image, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        image, labels = batch
        _, loss = self(image, labels)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        return {"val_loss": loss}

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        image, labels = batch
        logits, loss = self(image, labels)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=False)

        return {"logits": logits, "labels": labels}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        image, _ = batch
        logits = self(image)

        probabilities = torch.softmax(logits, dim=1)

        return {"logits": logits, "probabilities": probabilities}

    def configure_optimizers(self) -> Any:
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.conf["train"]["weight_decay"],
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.conf["train"]["learning_rate"],
        )
        lr_scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.conf["train"]["num_warmup_steps"],
            num_training_steps=self.conf["train"]["num_training_steps"],
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(pl.utilities.grad_norm(self, norm_type=2))
        super().on_before_optimizer_step(optimizer)
