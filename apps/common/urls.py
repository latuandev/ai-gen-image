from django.urls import path

from apps.common.views import ApiRootView, HealthCheckView

urlpatterns = [
    path('', ApiRootView.as_view(), name='api-v1-root'),
    path('health/', HealthCheckView.as_view(), name='health-check'),
]
