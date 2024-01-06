import fire
from omegaconf import OmegaConf

from hse_mlops_hw.infer import Inference
from hse_mlops_hw.train import Trainer


def train(config_path="../configs/config.yaml", version_base="1.3"):
    cfg = OmegaConf.load(config_path)
    cfg.version_base = version_base

    fire.Fire(Trainer(cfg).train)


def infer(config_path="../configs/config.yaml", version_base="1.3"):
    cfg = OmegaConf.load(config_path)
    cfg.version_base = version_base

    fire.Fire(Inference(cfg).infer)


if __name__ == "__main__":
    train()
    infer()
