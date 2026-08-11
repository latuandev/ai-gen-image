from typing import Any

from rest_framework import status, viewsets
from rest_framework.response import Response
from rest_framework.views import APIView


class APIResponseMixin:
    """
    Provide shared HTTP response helpers for API views and viewsets.
    """
    @staticmethod
    def response(data: Any = None, status_code: int = status.HTTP_200_OK) -> Response:
        """
        Return a DRF response with explicit data and HTTP status code.
        """
        return Response(data=data, status=status_code)


class BaseAPIView(APIResponseMixin, APIView):
    """
    Base class for APIView endpoints with shared response behavior.
    """


class BaseAPIViewSet(APIResponseMixin, viewsets.GenericViewSet):
    """
    Base class for ViewSet endpoints with shared response behavior.
    """


class BaseModelViewSet(APIResponseMixin, viewsets.ModelViewSet):
    """
    Base class for model-backed ViewSet endpoints with shared response behavior.
    """
