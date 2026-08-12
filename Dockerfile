FROM python:3.12-slim@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36 AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ARG PYTORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip wheel --no-cache-dir --no-deps --wheel-dir /wheels \
        --index-url "${PYTORCH_CPU_INDEX_URL}" torch==2.13.0 \
    && python -m pip wheel --no-cache-dir --wheel-dir /wheels --find-links /wheels .

FROM python:3.12-slim@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN useradd --create-home --uid 10001 app
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

WORKDIR /app
COPY configs ./configs
RUN mkdir -p /app/artifacts && chown app:app /app/artifacts
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"
CMD ["ac-cfr-web", "--host", "0.0.0.0", "--port", "8000", "--project-root", "/app"]
