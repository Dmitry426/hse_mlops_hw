# HSE MLops Project
This is the final project for Mlops course from [girafe.ai](https://github.com/girafe-ai)

## Goals
The main goal is to study new MLOps techniques.
To achieve this, I took the classic cat vs dog problem and,
with the help of various pretrained models from Hugging Face,
trained a neural network with high quality.
The stack includes Hydra, PyTorch, and Lightning.

### Development

To launch the application in the development environment,
there are options to run it directly using python and through
docker-compose.
In these run modes, it's possible to update the application code on the fly without restarting
(except when adding new dependencies).


#### Python Runner train

```bash
 poetry run train
```
#### Python Runner infer
```bash
 poetry run infer
```

### Docker train and inference
You can run train and infer in a configured docker lightning cuda environment .
In fact, it is the best way to utilize cuda since docker has preconfigured and tested environment

#### Build dev docker
Note: it might take some time since image is about 3gb. Considering you have
good internet connection so it will be roughly ~ 5-8 min long. This Docker build does
not load 'heavy' dependencies like (torch , lightning and cuda drivers ) we use
what is preconfigured and already present in
`pytorchlightning/pytorch_lightning:2.1.2-py3.10-torch2.1-cuda12.1.0 `.
For the best experience please keep synchronize deps in your pyproject with docker environment .
Torch source repo [here](https://download.pytorch.org/whl/cu121)
is configured in `pyproject.toml` as supplementary source so feel free to adjust cuda
versions according to your hardware .

 ```bash
make build_docker_dev
```
#### Docker Runner train

 ```bash
make infer
```

#### Docker Runner train

 ```bash
make train
```
You can add ARGUMENTS="foo bar" to the call to specify hydra command line arguments .
Hydra docs [here](https://hydra.cc/docs/advanced/hydra-command-line-flags/)

#### Docker Runner infer commands example
In order ro see hydra cli help along with current config

 ```bash
make infer ARGUMENTS="--help"

```

### Docker metrics runner
You can run local metrics(mlflow, TensorBoard ) via doker compose

This command will create a .env file from .env.example and build the containers with metrics services.

```bash
make build_docker_metrics
```

This command will run metrics dockerfiles

```bash
make run_metrics
```

This one is to stop metrics containers

```bash
make metrics_stop
```


#### Project linting:
You can lint project by running

```bash
make lint
```

### Before You Begin


```bash
make dev
```

This command to set up pre commit config in order to check your code before commit

### Dependency's

Dependency management is handled by the  `poetry`. \
The list of dependencies is in the `pyproject.toml` file. \
Instructions for setting up a poetry environment for
PyCharm can be found [here](https://www.jetbrains.com/help/pycharm/poetry.html).

To add a dependency, simply write `poetry add requests` and the utility
will automatically choose a version that does not conflict with current dependencies.. \
Dependencies with exact versions are recorded in the `poetry.lock`. \
To get a dependency tree, you can use the command  `poetry show --tree`.
Other commands are available in the official documentation for the utility.
