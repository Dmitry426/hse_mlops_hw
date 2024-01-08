import fire
from hydra import compose, initialize_config_dir

from hse_mlops_hw import PROJECT_ROOT
from hse_mlops_hw.infer import Inference
from hse_mlops_hw.train import Trainer


def infer(
    version_base: str = "1.3",
    config_path: str = str(PROJECT_ROOT / "configs"),
    job_name: str = "test_run",
    config_name: str = "config",
    overrides: list = None,
    return_hydra_config: bool = False,
):
    with initialize_config_dir(
        version_base=version_base, config_dir=config_path, job_name=job_name
    ):
        cfg = compose(
            config_name=config_name,
            overrides=overrides,
            return_hydra_config=return_hydra_config,
        )
        Inference(cfg).infer()


def train(
    version_base: str = "1.3",
    config_path: str = str(PROJECT_ROOT / "configs"),
    job_name: str = "test_run",
    config_name: str = "config",
    overrides: list = None,
    return_hydra_config: bool = False,
):
    with initialize_config_dir(
        version_base=version_base, config_dir=config_path, job_name=job_name
    ):
        cfg = compose(
            config_name=config_name,
            overrides=overrides,
            return_hydra_config=return_hydra_config,
        )
        Trainer(cfg).train()


if __name__ == "__main__":
    commands = {"infer": infer, "train": train}
    fire.Fire(commands)
