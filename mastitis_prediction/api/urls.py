from django.conf.urls import url
from django.views.decorators.csrf import csrf_exempt

from .views import ArticleListView, ArticleDetailsView, model_predict

urlpatterns = [
    url(r'^$', ArticleListView.as_view()),
    url(r'^(?P<pk>\d+)/$', ArticleDetailsView.as_view()),
    url(r'modelPredict/', csrf_exempt(model_predict))
]