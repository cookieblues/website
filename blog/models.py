from django.db import models
from django.utils.timezone import now


class Post(models.Model):
    class Meta:
        ordering = ["-date_created"]

    title = models.CharField(max_length=255, unique=True, primary_key=True)
    body = models.TextField()
    featured_image = models.TextField(default=None, null=True)
    date_created = models.DateField(default=now)
    date_modified = models.DateField(default=now)
    published = models.BooleanField(default=False)

    def __str__(self):
        return self.title
