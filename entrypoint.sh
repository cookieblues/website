#!/bin/sh

if [ "$DATABASE" = "postgres" ]
then
    echo "Waiting for postgres..."

    while ! nc -z $SQL_HOST $SQL_PORT; do
      sleep 0.1
    done

    echo "PostgreSQL started"
fi

poetry run python manage.py flush --no-input
poetry run python manage.py migrate
poetry run python manage.py collectstatic --no-input --clear
poetry run python manage.py runserver 0.0.0.0:8000 --insecure

exec "$@"
