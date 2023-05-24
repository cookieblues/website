from celery.utils.log import get_task_logger

from cookiesite.celery import app

logger = get_task_logger(__name__)


app.conf.beat_schedule = {
    "blog-test-task": {
        "task": "blog.tasks.add_test",
        "schedule": 30,
    },
}


@app.task(bind=True)
def add_test(self):
    logger.debug("test job")
    print("test jobby")
