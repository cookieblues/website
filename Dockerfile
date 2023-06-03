#################################
# BUILDER IMAGE
#################################
FROM python:3.9.16-alpine as builder-base

# install poetry
RUN pip install "poetry==1.4.2"

# set work directory
WORKDIR /home/app/web

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
COPY --from=builder-base /home/app/web/requirements.txt /home/app/web/requirements.txt

WORKDIR /home/app/web

RUN pip install --no-cache-dir -r /home/app/web/requirements.txt

# copy code
COPY /manage.py .
COPY /cookiesite /home/app/web/cookiesite
COPY /blog /home/app/web/blog

# copy entrypoints
COPY /entrypoint.sh /home/app/web/entrypoint.sh
RUN sed -i 's/\r$//g' /home/app/web/entrypoint.sh
RUN chmod +x /home/app/web/entrypoint.sh

COPY /worker-entrypoint.sh /home/app/web/worker-entrypoint.sh
RUN sed -i 's/\r$//g' /home/app/web/worker-entrypoint.sh
RUN chmod +x /home/app/web/worker-entrypoint.sh

COPY /beat-entrypoint.sh /home/app/web/beat-entrypoint.sh
RUN sed -i 's/\r$//g' /home/app/web/beat-entrypoint.sh
RUN chmod +x /home/app/web/beat-entrypoint.sh
