import re


def post_url_format(post_title: str) -> str:
    """Converts a title of a post into a proper URL string by removing any characters that are not letters,
    digits, or spaces. Lowercase the result and replace spaces with hyphens.
    """
    post_title = re.sub(r"[^a-zA-Z0-9 ]+", "", post_title)
    return post_title.lower().replace(" ", "-")
