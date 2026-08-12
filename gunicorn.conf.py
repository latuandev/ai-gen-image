import os

bind = "0.0.0.0:8000"

workers = int(os.getenv("GUNICORN_WORKERS", "2"))

timeout = int(os.getenv("GUNICORN_TIMEOUT", "60"))

reload = os.getenv("GUNICORN_RELOAD", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

accesslog = "-"
errorlog = "-"

capture_output = True
