# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"
COPY frontend/requirements.txt /src/requirements.txt
RUN pip install -r /src/requirements.txt

FROM python:3.12-slim AS runtime
ENV PATH="/venv/bin:$PATH" PYTHONUNBUFFERED=1
RUN useradd --create-home --uid 10001 app
COPY --from=builder /venv /venv
WORKDIR /app
COPY frontend/app.py .
USER app
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=3s --retries=10 CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:8501/_stcore/health'); sys.exit(0)"
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
