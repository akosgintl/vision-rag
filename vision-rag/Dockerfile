FROM python:3.12-slim

WORKDIR /app

# System deps for pdf2image + tini as init process
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils curl tini \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies (cached layer — only rebuilds when pyproject.toml changes)
COPY pyproject.toml .
COPY proxy/__init__.py proxy/__init__.py
RUN pip install --no-cache-dir .

# Copy full source
COPY proxy/ proxy/

# Non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --no-create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["curl", "-f", "http://localhost:8000/health"]

# tini handles signal forwarding and zombie reaping
ENTRYPOINT ["tini", "--"]
CMD ["uvicorn", "proxy.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
