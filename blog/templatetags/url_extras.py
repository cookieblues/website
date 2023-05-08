from django import template

import re

register = template.Library()


def post_url_format(post_title: str) -> str:
    post_title = re.sub(r"[^a-zA-Z0-9 ]+", "", post_title)
    return post_title.lower().replace(" ", "-")


@register.filter()
def space_to_hyphen(value):
    return post_url_format(value)
