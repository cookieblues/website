# pull official base image
FROM python:3.9.16-alpine

# install psycopg2 dependencies and poetry
RUN apk update && apk add postgresql-dev gcc python3-dev musl-dev
RUN pip install "poetry==1.4.2"

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# COPY /poetry.lock .
COPY /manage.py .
COPY /poetry.toml .
COPY /pyproject.toml .
COPY /README.md .

# install dependencies
RUN poetry install --no-interaction --no-ansi --without dev

# copy code
COPY /cookiesite /usr/src/app/cookiesite
COPY /blog /usr/src/app/blog

# copy entrypoints
COPY /entrypoint.sh .
RUN sed -i 's/\r$//g' /usr/src/app/entrypoint.sh
RUN chmod +x /usr/src/app/entrypoint.sh

COPY /worker-entrypoint.sh .
RUN sed -i 's/\r$//g' /usr/src/app/worker-entrypoint.sh
RUN chmod +x /usr/src/app/worker-entrypoint.sh

COPY /beat-entrypoint.sh .
RUN sed -i 's/\r$//g' /usr/src/app/beat-entrypoint.sh
RUN chmod +x /usr/src/app/beat-entrypoint.sh
