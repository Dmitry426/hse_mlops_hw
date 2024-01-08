.PHONY: dev pre-commit isort black mypy  pylint lint

dev: pre-commit

pre-commit:
	pre-commit install
	pre-commit autoupdate

isort:
	isort . --profile black

black:
	black .

mypy:
	mypy -p hse_mlops_hw

flake8:
	flake8 .

pylint:
	pylint hse_mlops_hw

lint: isort black mypy pylint flake8

check_and_rename_env:
	  @if [ -e ".env" ]; then \
        echo "env file exists."; \
      else \
      	cp .env.example .env | \
        echo "File does not exist."; \
      fi

build_docker_metrics: check_and_rename_env
	docker compose -f docker-compose.metrics.yaml build

build_docker_dev:check_and_rename_env
	docker compose -f docker-compose.dev.yaml build

run_metrics:
	docker compose -f docker-compose.metrics.yaml up

metrics_stop:
	docker compose -f docker-compose.metrics.yaml up

infer:
	docker compose -f docker-compose.dev.yaml run dev poetry run infer $(ARGUMENTS)

train:
	docker compose -f docker-compose.dev.yaml run dev poetry run train $(ARGUMENTS)
