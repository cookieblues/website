# Generated by Django 4.2.1 on 2023-05-21 14:05

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Post",
            fields=[
                (
                    "title",
                    models.CharField(
                        max_length=255, primary_key=True, serialize=False, unique=True
                    ),
                ),
                ("body", models.TextField()),
                ("featured_image", models.TextField(default=None, null=True)),
                ("date_created", models.DateField(default=django.utils.timezone.now)),
                ("date_modified", models.DateField(default=django.utils.timezone.now)),
                ("published", models.BooleanField(default=False)),
            ],
            options={
                "ordering": ["-date_created"],
            },
        ),
    ]
