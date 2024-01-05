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
    	chmod +r .env \
    	. .env | \
        echo "env file exists."; \
      else \
      	cp .env.example .env | \
      	chmod +r .env | \
      	. .env | \
        echo "File does not exist."; \
      fi
