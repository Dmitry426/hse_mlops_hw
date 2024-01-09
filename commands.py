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
) -> None:
    """
    Run inference using a pre-trained model based on the specified configuration.

    Args:
        version_base (str, optional): Hydra config version. Defaults to "1.3".
        config_path (str, optional): Path to the configuration directory.
        Defaults to "PROJECT_ROOT/configs".
        job_name (str, optional): Job name for the configuration.
        Defaults to "test_run".
        config_name (str, optional): Name of the configuration file.
         Defaults to "config".
        overrides (List[str], optional): List of overrides for the Hydra configuration.
         Defaults to None.
        return_hydra_config (bool, optional): Whether to return the Hydra config.
         Defaults to False.

    Returns:
        None

    Note:
       This function uses Hydra for configuration management.
       It initializes the configuration directory,
       composes the configuration based on the provided parameters,
       and then trains a model using the PyTorch Lightning Trainer.

    """
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
) -> None:
    """
    Train a model based on the specified configuration using PyTorch Lightning.

    Args:
        version_base (str, optional): Hydra config version.
        Defaults to "1.3".
        config_path (str, optional): Path to the configuration directory.
        Defaults to "PROJECT_ROOT/configs".
        job_name (str, optional): Job name for the configuration.
         Defaults to "test_run".
        config_name (str, optional): Name of the configuration file.
        Defaults to "config".
        overrides (List[str], optional): List of overrides for the Hydra configuration.
         Defaults to None.
        return_hydra_config (bool, optional): Whether to return the Hydra config.
        Defaults to False.

    Returns:
        None

    Note:
        This function uses Hydra for configuration management.
        It initializes the configuration directory,
        composes the configuration based on the provided parameters,
        and then trains a model using the PyTorch Lightning Trainer.

    """
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
