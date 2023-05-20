import re


def post_url_format(post_title: str) -> str:
    post_title = re.sub(r"[^a-zA-Z0-9 ]+", "", post_title)
    return post_title.lower().replace(" ", "-")
