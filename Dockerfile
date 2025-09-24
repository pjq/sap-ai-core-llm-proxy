# Lightweight Python image
FROM python:3.11-slim

# Avoids prompts from some apt operations
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system deps needed for building some Python packages and SSL
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Install dependencies first (for better caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Expose default port
EXPOSE 3001

# Default configuration path can be overridden with CONFIG_PATH env var
ENV CONFIG_PATH=/app/config.json \
    HOST=0.0.0.0 \
    PORT=3001

# For SAP AI SDK, ~/.aicore/config.json should be mounted into /root/.aicore/config.json
# Example: -v $HOME/.aicore:/root/.aicore:ro

# Healthcheck (basic)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://localhost:${PORT}/v1/models || exit 1

# Start the proxy server (host/port can be changed via env)
CMD ["/bin/sh", "-lc", "python proxy_server.py --config ${CONFIG_PATH}${DEBUG:+ --debug} --debug"]
