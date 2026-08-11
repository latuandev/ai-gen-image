"""
URL configuration for core project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path, include, re_path

from apps.common.views import ApiRootView, NotFoundView

api_v1_patterns = [
    path('', include('apps.common.urls')),
]

urlpatterns = [
    path('', ApiRootView.as_view(), name='api-root'),
    path('api/v1/', include(api_v1_patterns)),
    re_path(r'^(?P<path>.*)$', NotFoundView.as_view(), name='api-not-found'),
]
