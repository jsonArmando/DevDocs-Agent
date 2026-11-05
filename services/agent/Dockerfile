# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"
COPY libs/core /src/libs/core
COPY services/agent/requirements.txt /src/requirements.txt
RUN pip install /src/libs/core -r /src/requirements.txt

FROM python:3.12-slim AS runtime
ENV PATH="/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN useradd --create-home --uid 10001 app && mkdir -p /models /data && chown app /models /data
COPY --from=builder /venv /venv
WORKDIR /app
COPY services/agent/app ./app
USER app
EXPOSE 8002
HEALTHCHECK --interval=15s --timeout=3s --retries=10 CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:8002/health'); sys.exit(0)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
