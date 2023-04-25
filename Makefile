PYTHON_VERSION = 3.9.16
ACTIVATE_ENV = . .venv/bin/activate

.PHONY: dev-setup
dev-setup:
	pyenv install -s $(PYTHON_VERSION)
	pyenv local $(PYTHON_VERSION)
	poetry config virtualenvs.in-project true --local
	poetry env use $(PYTHON_VERSION)
	poetry install
	$(ACTIVATE_ENV) && pre-commit install -c .pre-commit-config.yaml

.PHONY: clean
clean:
	rm -rf .venv
	rm poetry.lock

.PHONY: black
black:
	$(ACTIVATE_ENV) && poetry run black .

.PHONY: isort
isort:
	$(ACTIVATE_ENV) && poetry run isort .

.PHONY: flake8
flake8:
	$(ACTIVATE_ENV) && poetry run flake8 .

.PHONY: mypy
mypy:
	$(ACTIVATE_ENV) && poetry run mypy .
