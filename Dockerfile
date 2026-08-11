FROM python:3.14.7-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt


# =========================================================
# Development
# =========================================================
FROM base AS development

COPY . .

EXPOSE 8000


# =========================================================
# Production
# =========================================================
FROM base AS production

COPY . .

RUN addgroup --system ai-gen-image \
    && adduser --system --ingroup ai-gen-image ai-gen-image \
    && chown -R ai-gen-image:ai-gen-image /app

USER ai-gen-image

EXPOSE 8000
