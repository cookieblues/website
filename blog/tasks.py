from celery.schedules import crontab
from celery.task import periodic_task
from celery.utils.log import get_task_logger

from blog.models import Post

logger = get_task_logger(__name__)


@periodic_task(run_every=(crontab()), name="random_test", ignore_result=True)
def add_test():
    new_post = Post(
        title="test",
        body="test body",
        featured_image=None,
        published=True,
    )
    Post.objects.create(new_post)
