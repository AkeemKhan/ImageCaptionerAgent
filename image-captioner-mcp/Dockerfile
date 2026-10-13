# Image captioning MCP server over streamable HTTP.
# GPU build (default):  docker build -t image-captioner-mcp .
# CPU build:            docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu -t image-captioner-mcp .
FROM python:3.11-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    CAPTIONER_ALLOWED_ROOT=/data/sessions

WORKDIR /app
RUN pip install torch torchvision --index-url ${TORCH_INDEX}
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[florence]"

RUN useradd --create-home app && mkdir -p /data/sessions /models && chown -R app /data /models
USER app

EXPOSE 8765
# 0.0.0.0 so other containers can reach it; publish the port only to hosts you trust.
CMD ["image-captioner-mcp", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8765"]
