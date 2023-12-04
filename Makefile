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
	mypy -p src

pylint:
	pylint src

lint: isort black mypy  pylint

