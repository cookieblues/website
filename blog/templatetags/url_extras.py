from django import template

from blog.utils import post_url_format


register = template.Library()


@register.filter()
def space_to_hyphen(value):
    return post_url_format(value)
