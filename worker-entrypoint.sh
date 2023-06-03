#!/bin/sh

until cd /home/app/web/
do
    echo "Waiting for server volume..."
done

# run a worker
celery -A cookiesite worker --loglevel=info --concurrency 1 -E
