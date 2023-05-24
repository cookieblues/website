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

.PHONY: build-docker
build-docker:
	docker-compose down -v --remove-orphans
	docker-compose build
	docker-compose up -d
	docker-compose exec web python manage.py collectstatic --no-input --clear
	docker image prune -af

.PHONY: check-docker-logs
check-docker-logs:
	docker-compose logs -f

.PHONY: push-stage
push-stage:
	docker-compose down -v
	docker-compose -f docker-compose.stage.yml build
	aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin 284110347794.dkr.ecr.eu-west-1.amazonaws.com
	docker-compose -f docker-compose.stage.yml push

.PHONY: copy-files-to-instance
copy-files-to-instance:
	scp -i ~/.ssh/djangoletsencrypt.pem -r $(pwd)/{cookiesite,blog,nginx,.env.stage,.env.stage.proxy-companion,docker-compose.stage.yml,Dockerfile.prod,entrypoint.prod.sh,manage.py,poetry.toml,pyproject.toml,README.md} ubuntu@34.250.69.75:/home/ubuntu/cookieblues-website-stage
