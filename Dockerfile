# pull official base image
FROM python:3.9.16-alpine

RUN pip install "poetry==1.4.2"

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY /cookiesite /usr/src/app/cookiesite
# COPY /poetry.lock .
COPY /poetry.toml .
COPY /pyproject.toml .
COPY /README.md .

# install dependencies
RUN poetry install --no-interaction --no-ansi --without dev
# RUN pip install --upgrade pip
# COPY ./requirements.txt .
# RUN pip install -r requirements.txt

# # copy project
# COPY . .
