from django.shortcuts import render
from django.views import View

from blog.models import Post
from blog.utils import post_url_format


class Main(View):
    def __init__(self):
        self.context = {"include_sidebar": True}


class Frontpage(Main):
    def __init__(self):
        super().__init__()
        self.template = "blog/index.html"

    def get(self, request):
        self.context["posts"] = Post.objects.filter(published=True).values()
        for post in self.context["posts"]:
            post["excerpt"], _ = post["body"].split("<!--more-->")
        return render(request, self.template, self.context)


class Postpage(Main):
    def __init__(self):
        super().__init__()
        self.template = "blog/post.html"

    def get(self, request, post_url):
        posts = Post.objects.all()
        post_urls = list(map(lambda x: post_url_format(x.title), posts))
        post_idx = post_urls.index(post_url)
        self.context["post"] = posts[post_idx]
        return render(request, self.template, self.context)
