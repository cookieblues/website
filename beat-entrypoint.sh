#!/bin/sh

until cd /usr/src/app
do
    echo "Waiting for server volume..."
done

# run the beater
poetry run celery -A cookiesite beat --loglevel=info
