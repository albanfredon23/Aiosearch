# syntax=docker/dockerfile:1.7
FROM python:3.13-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

FROM base AS build
COPY pyproject.toml ./
COPY aiotech ./aiotech
RUN --mount=type=secret,id=build_ca,required=false \
    if [ -s /run/secrets/build_ca ]; then export PIP_CERT=/run/secrets/build_ca; fi \
    && pip wheel --wheel-dir /wheels ".[mcp,providers]"

FROM base AS test
RUN --mount=type=bind,from=build,source=/wheels,target=/wheels \
    --mount=type=secret,id=build_ca,required=false \
    if [ -s /run/secrets/build_ca ]; then export PIP_CERT=/run/secrets/build_ca; fi \
    && pip install --no-index --find-links /wheels "aiotech-search[mcp]" \
    && pip install "pytest>=8.0" "pytest-asyncio>=0.24" "fakeredis>=2.20"
COPY pyproject.toml ./
COPY tests ./tests
CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]

FROM base AS runtime
RUN groupadd --gid 10001 aiotech \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin aiotech \
    && mkdir -p /run/aiotech /data/corpus /data/docs /out \
    && chown aiotech:aiotech /run/aiotech /data/corpus /out
RUN --mount=type=bind,from=build,source=/wheels,target=/wheels \
    pip install --no-index --find-links /wheels "aiotech-search[mcp,providers]"
USER aiotech
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=4s --start-period=20s --retries=5 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status != 200)"]
CMD ["uvicorn", "--factory", "aiotech.api.app:create_app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*", "--no-server-header"]
