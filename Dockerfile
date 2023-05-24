#################################
# BUILDER IMAGE
#################################
FROM python:3.9.16-alpine as builder-base

# install poetry
RUN pip install "poetry==1.4.2"

# set work directory
WORKDIR /usr/src/app

COPY /poetry.toml .
COPY /pyproject.toml .
COPY /README.md .

# install dependencies
RUN poetry install --no-cache --no-interaction --no-ansi --without dev
RUN poetry export -f requirements.txt >> requirements.txt


#################################
# PRODUCTION IMAGE
#################################
FROM python:3.9.16-alpine
ENV PYTHONDONTWRITEBYTECODE 1 \
    PYTHONUNBUFFERED 1 \
    PIP_NO_CACHE_DIR=off
COPY --from=builder-base /usr/src/app/requirements.txt /usr/src/app/requirements.txt

WORKDIR /usr/src/app

RUN pip install --no-cache-dir -r /usr/src/app/requirements.txt

# copy code
COPY /manage.py .
COPY /cookiesite /usr/src/app/cookiesite
COPY /blog /usr/src/app/blog

# copy entrypoints
COPY /entrypoint.sh /usr/src/app/entrypoint.sh
RUN sed -i 's/\r$//g' /usr/src/app/entrypoint.sh
RUN chmod +x /usr/src/app/entrypoint.sh

COPY /worker-entrypoint.sh /usr/src/app/worker-entrypoint.sh
RUN sed -i 's/\r$//g' /usr/src/app/worker-entrypoint.sh
RUN chmod +x /usr/src/app/worker-entrypoint.sh

COPY /beat-entrypoint.sh /usr/src/app/beat-entrypoint.sh
RUN sed -i 's/\r$//g' /usr/src/app/beat-entrypoint.sh
RUN chmod +x /usr/src/app/beat-entrypoint.sh
