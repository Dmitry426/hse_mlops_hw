FROM pytorchlightning/pytorch_lightning:2.1.2-py3.10-torch2.1-cuda12.1.0 as base

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PYTHONDONTWRITEBYTECODE=1 \
    # pip:
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    # poetry:
    POETRY_VERSION=1.7.1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR='/var/cache/pypoetry' \
    PATH="$PATH:/root/.local/bin"

RUN mkdir /src
RUN mkdir /src/hse_mlops_hw

WORKDIR /src

RUN apt-get update && apt-get install --no-install-recommends -y \
    curl \
    build-essential

RUN curl -sSL https://install.python-poetry.org | python -

COPY pyproject.toml poetry.lock ./

COPY hse_mlops_hw ./src/hse_mlops_hw


RUN poetry install --no-dev  --no-interaction --no-ansi


CMD ["tail", "-f", "/dev/null"]
