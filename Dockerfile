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

# copy entrypoint.sh
RUN sed -i 's/\r$//g' /usr/src/app/cookiesite/entrypoint.sh
RUN chmod +x /usr/src/app/cookiesite/entrypoint.sh

# run entrypoint.sh
ENTRYPOINT ["/usr/src/app/cookiesite/entrypoint.sh"]
