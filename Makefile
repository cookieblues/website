DJANGO_DIR = django-app
VENV_DIR = $(DJANGO_DIR)/.venv
ACTIVATE_DJANGO_APP_VENV = . $(VENV_DIR)/bin/activate

dev-setup: pre-commit install
	cd $(DJANGO_DIR) && poetry install

black:
	$(ACTIVATE_DJANGO_APP_VENV) && cd $(DJANGO_DIR) && poetry run black .

isort:
	$(ACTIVATE_DJANGO_APP_VENV) && cd $(DJANGO_DIR) && poetry run isort .

flake8:
	$(ACTIVATE_DJANGO_APP_VENV) && cd $(DJANGO_DIR) && poetry run flake8 .

mypy:
	$(ACTIVATE_DJANGO_APP_VENV) && cd $(DJANGO_DIR) && poetry run mypy .
