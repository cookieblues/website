from django.shortcuts import render
from django.views import View
from django.http.response import HttpResponseRedirectBase


class Main(View):
    def __init__(self) -> None:
        self.context = {"include_sidebar": True}


class Frontpage(Main):
    def __init__(self):
        super().__init__()
        self.template = "blog/index.html"

    def get(self, request) -> HttpResponseRedirectBase:
        return render(request, self.template, self.context)
