from django.urls import path

from fish_services.views import HighlightingMomentsView, SurfaceWarningsView, \
    NewVideoProcessingView


urlpatterns = [
    path('surface-warnings/', SurfaceWarningsView.as_view()),
    path("highlighting-moments/", HighlightingMomentsView.as_view()),
    path("new-video/", NewVideoProcessingView.as_view()),
]
