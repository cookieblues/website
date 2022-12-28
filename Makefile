DJANGO_DIR = django-app
VENV_DIR = $(DJANGO_DIR)/.venv
ACTIVATE_DJANGO_APP_VENV = . $(VENV_DIR)/bin/activate

dev-setup: pre-commit install
	cd $(DJANGO_DIR) && poetry install

black:
	$(ACTIVATE_DJANGO_APP_VENV) && cd $(DJANGO_DIR) && poetry run black .

isort:
	$(VENV_DIR)/bin/pre-commit run isort --all-files

flake8:
	$(VENV_DIR)/bin/pre-commit run flake8 --all-files

mypy:
	$(VENV_DIR)/bin/pre-commit run mypy --all-files
