#!/bin/sh

until cd /home/app/web/
do
    echo "Waiting for server volume..."
done

# run the beater
celery -A cookiesite beat --loglevel=info
