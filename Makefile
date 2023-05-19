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
	$(ACTIVATE_ENV) && poetry run black cookiesite

.PHONY: isort
isort:
	$(ACTIVATE_ENV) && poetry run isort cookiesite

.PHONY: flake8
flake8:
	$(ACTIVATE_ENV) && poetry run flake8 cookiesite

.PHONY: mypy
mypy:
	$(ACTIVATE_ENV) && poetry run mypy cookiesite

.PHONY: build-docker
build-docker:
	docker-compose down -v
	docker-compose build
	docker-compose up -d

.PHONY: check-docker-logs
check-docker-logs:
	docker-compose logs -f

.PHONY: stop-container
stop-container:
	docker-compose down -v

.PHONY: run-docker-migrations
run-docker-migrations:
	docker-compose exec web poetry run python cookiesite/manage.py migrate --noinput

.PHONY: flush-and-migrate
flush-and-migrate:
	docker-compose exec web poetry run python manage.py flush --no-input
	docker-compose exec web poetry run python manage.py migrate

.PHONY: check-docker-db
check-docker-db:
	docker volume inspect website_postgres_data

.PHONY: rerun-prod
rerun-prod:
	docker-compose -f docker-compose.stage.yml down -v
	docker-compose -f docker-compose.stage.yml up -d --build
	docker-compose -f docker-compose.stage.yml exec web poetry run python manage.py migrate --noinput
	docker-compose -f docker-compose.stage.yml exec web poetry run python manage.py collectstatic --no-input --clear

.PHONY: build-and-push-stage
build-and-push-stage:
	docker-compose down -v
	docker-compose -f docker-compose.stage.yml build
	aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin 284110347794.dkr.ecr.eu-west-1.amazonaws.com
	docker-compose -f docker-compose.stage.yml push

.PHONY: copy-files-to-instance
copy-files-to-instance:
	scp -i ~/.ssh/djangoletsencrypt.pem -r $(pwd)/{cookiesite,blog,nginx,.env.stage,.env.stage.proxy-companion,docker-compose.stage.yml,Dockerfile.prod,entrypoint.prod.sh,manage.py,poetry.toml,pyproject.toml,README.md} ubuntu@34.250.69.75:/home/ubuntu/cookieblues-website-stage
