from rest_framework import status

from common.responses import BaseAPIView


class ApiRootView(BaseAPIView):
    """
    Expose API metadata and discoverable shared endpoints.
    """

    authentication_classes = []
    permission_classes = []

    def get(self, request):
        """
        Return API metadata and shared endpoint links.
        """
        return self.response(
            {
                "name": "AI Gen Image",
                "version": "v1",
                "endpoints": {
                    "health": "/api/v1/health/",
                },
            }
        )


class HealthCheckView(BaseAPIView):
    """
    Expose a lightweight endpoint for service health checks.
    """

    authentication_classes = []
    permission_classes = []

    def get(self, request):
        """
        Return a lightweight health status for service checks.
        """
        return self.response({"status": "OK"})


class NotFoundView(BaseAPIView):
    """
    Render unmatched routes as JSON API 404 responses.
    """

    authentication_classes = []
    permission_classes = []

    def get(self, request, path=None):
        """
        Return a JSON 404 response for unmatched API routes.
        """
        return self.response(
            {
                "detail": "API not found",
                "path": f"/{path or ''}",
            },
            status_code=status.HTTP_404_NOT_FOUND,
        )

    post = put = patch = delete = get
