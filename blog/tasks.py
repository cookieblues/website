from celery.schedules import crontab
from celery.task import periodic_task
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)


@periodic_task(run_every=(crontab()), name="random_test", ignore_result=True)
def add_test():
    logger.debug("test job")
    print("test jobby")
