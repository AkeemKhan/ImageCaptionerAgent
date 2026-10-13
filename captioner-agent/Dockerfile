# Web chat agent. Expects the MCP server to be reachable over HTTP (see docker-compose.yml).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CAPTIONER_AGENT_SETTINGS=/app/config/settings.docker.yaml

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .
COPY config ./config

RUN useradd --create-home app && mkdir -p /data/sessions && chown -R app /data
USER app

EXPOSE 8000
CMD ["captioner-agent"]
