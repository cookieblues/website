from django.test import TestCase

from blog.utils import post_url_format


class RandomName(TestCase):
    def test_post_url_format(self):
        assert post_url_format("This is an example of a Title Post") == "this-is-an-example-of-a-title-post"
